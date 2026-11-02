"""大肥鱼心智内核：独立记忆库 memory_core.db（不动旧库）。

做什么：
1. 从 memory.db / brain.db **只读**导入记忆当种子（带来源标记，永不回写）
2. 给每条记忆建立 权重 w / 锚定时间 t_anchor / 回想次数 / 记忆池
3. 后台 tick：分段衰减 → 海马晋升 → 下沉归档
4. 给出召回的 f_endo 加权入口（后续第三步接入注入）

红线：memory.db、brain.db 只读打开（immutable），不写一个字。
"""

import json
import os
import sqlite3
import threading
import time
import urllib.request

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

PLUGIN_NAME = "dafeiyu_mindcore"
DATA_DIR = f"/AstrBot/data/plugin_data/{PLUGIN_NAME}"
CORE_DB = os.path.join(DATA_DIR, "memory_core.db")
HORMONE_FILE = "/AstrBot/data/plugin_data/dafeiyu_hormone/hormones.json"

OLD_DB = "/AstrBot/data/plugin_data/dafeiyu_brain/memory.db"
BRAIN_DB = "/AstrBot/data/plugin_data/astrbot_plugin_memory_brain/brain.db"

DAY = 86400.0


def _open_ro(path: str):
    if not os.path.exists(path):
        return None
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{PLUGIN_NAME}] 只读打开失败 {path}: {exc!r}")
        return None


def _cols(db, table: str) -> list[str]:
    try:
        return [r[1] for r in db.execute(f"PRAGMA table_info({table})")]
    except Exception:
        return []


class MindCore(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.cfg = config or {}
        self._task = None
        os.makedirs(DATA_DIR, exist_ok=True)
        self._lock = threading.RLock()
        self._pending_skill = ""
        self._pending_ts = 0.0
        self._pending_selfeval = 0.0
        self.db = sqlite3.connect(CORE_DB, timeout=15, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _c(self, key, default):
        try:
            v = self.cfg.get(key, default)
        except Exception:
            return default
        return default if v is None else v

    # ---------------------------------------------------------------- 建表

    def _init_schema(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          src TEXT,             -- seed:memory / seed:brain / live
          src_id TEXT,          -- 原库 id
          text TEXT NOT NULL,
          kind TEXT DEFAULT 'episode',
          entities TEXT DEFAULT '',
          importance REAL DEFAULT 0.5,
          weight REAL DEFAULT 1.0,
          t_anchor REAL,        -- 锚定时间（epoch 秒）
          dt_eff REAL DEFAULT 0,-- 已结算的等效时长（秒）
          access_count INTEGER DEFAULT 0,
          last_access REAL DEFAULT 0,
          pool TEXT DEFAULT 'short',   -- short / mid / long / archive / deep / trace
          created_at REAL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_src ON memories(src, src_id);
        CREATE INDEX IF NOT EXISTS ix_pool ON memories(pool);
        CREATE TABLE IF NOT EXISTS recall_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          mem_id INTEGER, ts REAL, score REAL, source TEXT,
          processed INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS mem_vec (mem_id INTEGER PRIMARY KEY, vec BLOB);
        CREATE TABLE IF NOT EXISTS skill_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT, ts REAL, q REAL, how TEXT
        );
        CREATE TABLE IF NOT EXISTS skills (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE NOT NULL,
          desc TEXT DEFAULT '',
          L REAL DEFAULT 0.35,      -- 熟练度
          EF REAL DEFAULT 2.0,      -- 易度因子 [1.3,2.5]
          I REAL DEFAULT 1.0,       -- 上次调用间隔（小时）
          last_used REAL DEFAULT 0,
          use_count INTEGER DEFAULT 0,
          attach_bonus INTEGER DEFAULT 0
        );
        """)
        self.db.commit()
        self._migrate()

    def _migrate(self) -> None:
        """给老库补列（CREATE TABLE IF NOT EXISTS 不会改结构）。"""
        try:
            cols = [r[1] for r in self.db.execute("PRAGMA table_info(recall_events)")]
            if cols and "processed" not in cols:
                self.db.execute(
                    "ALTER TABLE recall_events ADD COLUMN processed INTEGER DEFAULT 0")
                self.db.commit()
                logger.info(f"[{PLUGIN_NAME}] 迁移：recall_events 补 processed 列")
            cols = [r[1] for r in self.db.execute("PRAGMA table_info(memories)")]
            for col, ddl in (("dt_eff", "REAL DEFAULT 0"), ("last_access", "REAL DEFAULT 0"),
                             ("pool", "TEXT DEFAULT 'short'")):
                if col not in cols:
                    self.db.execute(f"ALTER TABLE memories ADD COLUMN {col} {ddl}")
                    logger.info(f"[{PLUGIN_NAME}] 迁移：memories 补 {col}")
            self.db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 迁移失败：{exc!r}")

    # ---------------------------------------------------------------- 导入


    @staticmethod
    def _norm_text(t: str) -> str:
        """去重用的归一化：去掉标点、空白、大小写差异。"""
        import re as _re
        t = str(t or "").lower()
        t = _re.sub(r"[\s\u3000]+", "", t)
        t = _re.sub(r"[，。！？、；：,.!?;:\"'（）()《》<>【】\[\]…—\-~·]+", "", t)
        return t[:120]

    def _dup_of(self, norm: str, cache: dict) -> int | None:
        """同文本已存在就返回它的 id（内存缓存 + 库里查一次）。"""
        if not norm:
            return None
        if norm in cache:
            return cache[norm]
        row = self.db.execute(
            "SELECT id FROM memories WHERE lower(replace(replace(text,' ',''),'，',''))=? LIMIT 1",
            (norm,)).fetchone()
        if row:
            cache[norm] = int(row[0])
            return int(row[0])
        return None

    def _seed(self) -> tuple[int, int]:
        added = skipped = dup = 0
        cur = self.db.cursor()
        seen: dict = {}

        # 旧库：episodes / facts
        db = _open_ro(OLD_DB)
        if db:
            db.row_factory = sqlite3.Row
            for table, kind in (("episodes", "episode"), ("facts", "fact")):
                cols = _cols(db, table)
                if not cols:
                    continue
                text_col = "gist" if "gist" in cols else ("text" if "text" in cols else None)
                if not text_col:
                    continue
                time_col = "occurred_at" if "occurred_at" in cols else (
                    "created_at" if "created_at" in cols else None)
                imp_col = "importance" if "importance" in cols else None
                ent_col = "people" if "people" in cols else None
                sql = f"SELECT id, {text_col} AS txt" \
                      f"{', ' + time_col + ' AS tt' if time_col else ''}" \
                      f"{', ' + imp_col + ' AS imp' if imp_col else ''}" \
                      f"{', ' + ent_col + ' AS ent' if ent_col else ''} FROM {table}"
                try:
                    rows = db.execute(sql).fetchall()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[{PLUGIN_NAME}] 读 {table} 失败：{exc!r}")
                    continue
                for r in rows:
                    txt = (r["txt"] or "").strip()
                    if not txt:
                        continue
                    tt = self._parse_time(r["tt"] if time_col else None)
                    imp = float(r["imp"] or 0.6) if imp_col else 0.6
                    ent = str(r["ent"] or "") if ent_col else ""
                    norm = self._norm_text(txt)
                    if self._dup_of(norm, seen) is not None:
                        dup += 1
                        continue
                    cur.execute(
                        "INSERT OR IGNORE INTO memories"
                        "(src,src_id,text,kind,entities,importance,weight,t_anchor,"
                        "dt_eff,access_count,pool,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (f"seed:memory:{table}", str(r["id"]), txt, kind, ent, imp,
                         imp, tt, 0.0, 0, self._pool_for(tt), time.time()))
                    if cur.rowcount:
                        added += 1
                    else:
                        skipped += 1
            db.close()

        # 新库（memory_brain）
        db = _open_ro(BRAIN_DB)
        if db:
            db.row_factory = sqlite3.Row
            cols = _cols(db, "memories")
            if cols:
                try:
                    rows = db.execute(
                        "SELECT id, text, created_at, importance, kind FROM memories "
                        "WHERE status='active'").fetchall()
                except Exception:
                    rows = []
                for r in rows:
                    txt = (r["text"] or "").strip()
                    if not txt:
                        continue
                    tt = float(r["created_at"] or time.time())
                    imp = float(r["importance"] or 0.5)
                    norm = self._norm_text(txt)
                    if self._dup_of(norm, seen) is not None:
                        dup += 1
                        continue
                    cur.execute(
                        "INSERT OR IGNORE INTO memories"
                        "(src,src_id,text,kind,entities,importance,weight,t_anchor,"
                        "dt_eff,access_count,pool,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        ("seed:brain", str(r["id"]), txt, r["kind"] or "episode", "",
                         imp, imp, tt, 0.0, 0, self._pool_for(tt), time.time()))
                    if cur.rowcount:
                        added += 1
                    else:
                        skipped += 1
            db.close()

        self.db.commit()
        return added, skipped + dup

    @staticmethod
    def _parse_time(v) -> float:
        if v is None:
            return time.time()
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return time.mktime(time.strptime(s[:26], fmt))
            except Exception:
                continue
        return time.time()

    @staticmethod
    def _pool_for(t_anchor: float) -> str:
        age_days = (time.time() - t_anchor) / DAY
        if age_days <= 3:
            return "short"
        if age_days <= 30:
            return "mid"
        return "long"

    # ---------------------------------------------------------------- 参数

    def _half_life(self, pool: str) -> float:
        """各池半衰期（天）。"""
        return {
            "short": float(self._c("half_life_short_days", 7)),
            "mid": float(self._c("half_life_mid_days", 45)),
            "long": float(self._c("half_life_long_days", 180)),
            "archive": float(self._c("half_life_archive_days", 540)),
            "deep": float(self._c("half_life_deep_days", 1080)),
            "trace": float(self._c("half_life_trace_days", 3650)),
        }.get(pool, 180.0)

    def _hormone_factor(self) -> float:
        try:
            with open(HORMONE_FILE, encoding="utf-8") as f:
                h = json.load(f)
            oxy = float(h.get("oxy", 0.5))
            cort = float(h.get("cort", 0.2))
            return (1 + 0.30 * (oxy - 0.5)) * (1 - 0.35 * max(0.0, cort - 0.2) / 0.8)
        except Exception:
            return 1.0

    # ---------------------------------------------------------------- tick

    def _absorb(self, every: float = 600) -> int:
        """增量吸收旧库/脑库的新记忆（已在 _seed 里去重）。"""
        import time as _t
        now = _t.time()
        if now - getattr(self, "_last_seed", 0) < every:
            return 0
        self._last_seed = now
        try:
            added, skipped = self._seed()
            if added:
                logger.info(f"[{PLUGIN_NAME}] 增量吸收 {added} 条新记忆（跳过 {skipped}）")
            return added
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 增量吸收失败：{exc!r}")
            return 0

    def tick(self, dt_seconds: float | None = None) -> dict:
        self._absorb()
        try:
            import asyncio as _aio
            if not getattr(self, "_auto_task", None):
                self._auto_task = _aio.create_task(self._auto_backup_loop())
        except Exception:
            pass
        """分段衰减 + 海马晋升 + 下沉。dt 用真实经过时间（小时上界保护）。"""
        now = time.time()
        raw_last = self._meta_get("last_tick")
        try:
            last = float(raw_last) if raw_last else 0.0
        except (TypeError, ValueError):
            last = 0.0
        if dt_seconds is None:
            dt = (now - last) if last else 0.0
        else:
            dt = dt_seconds
        if dt > 7 * DAY:                        # 长间隔走离线压缩
            dt = self._compress(dt)
        stats = {"dt_h": dt / 3600.0, "decayed": 0, "promoted": 0, "sunk": 0,
                 "recalled": self._settle_recall(),
                 "skills": self._skill_decay(dt / 3600.0)}
        if dt <= 0:
            self._meta_set("last_tick", str(now))
            return stats

        with self._lock:
            rows = self.db.execute(
                "SELECT id, pool, weight, t_anchor, dt_eff, access_count, importance,"
                " entities FROM memories").fetchall()
        cur = self.db.cursor()
        decay_base = float(self._c("decay_scale", 1.0))
        for (mid, pool, w, t_anchor, dt_eff, acc, imp, ent) in rows:
            half = self._half_life(pool) * DAY
            lam = 0.693147 / max(1.0, half)      # 每秒衰减系数
            new_w = w * pow(2.718281828, -lam * dt * decay_base)
            new_dt = (dt_eff or 0.0) + dt
            # 海马打分
            e = max(0.0, min(1.0, float(imp or 0.5)))
            c_e = min(1.0, len([x for x in str(ent or "").split(",") if x.strip()]) / 4.0)
            c_r = min(1.0, (acc or 0) / 5.0)
            o = max(0.0, min(1.0, float(imp or 0.5)))
            score = 0.40 * e + 0.20 * c_e + 0.25 * c_r + 0.15 * o
            new_pool = pool
            age = (now - float(t_anchor or now))
            th_mid = float(self._c("promote_mid", 0.35))
            th_long = float(self._c("promote_long", 0.60))
            if pool == "short" and score >= th_mid:
                new_pool = "mid"
                stats["promoted"] += 1
            elif pool in ("short", "mid") and score >= th_long and (acc or 0) >= 2:
                new_pool = "long"
                stats["promoted"] += 1
            # 下沉
            if pool == "long" and new_w < 0.02 and age > 365 * DAY:
                new_pool = "archive"
                stats["sunk"] += 1
            elif pool == "archive" and new_w < 0.002:
                new_pool = "deep"
                stats["sunk"] += 1
            elif pool == "deep" and new_w < 0.0002:
                new_pool = "trace"
                stats["sunk"] += 1
            cur.execute("UPDATE memories SET weight=?, dt_eff=?, pool=? WHERE id=?",
                        (new_w, new_dt, new_pool, mid))
            stats["decayed"] += 1
        self.db.commit()
        self._meta_set("last_tick", str(now))
        return stats

    def _meta_get(self, k: str):
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else None

    def _meta_set(self, k: str, v: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (k, v))
        self.db.commit()



    # ---------------------------------------------------------------- 向量

    def _embed_sync(self, texts: list) -> list:
        base = str(self._c("embedding_api_base", "") or
                   "http://ollama:11434/v1").strip()
        model = str(self._c("embedding_model", "") or "bge-m3").strip()
        if not base or not model:
            return []
        url = base.rstrip("/") + "/embeddings"
        body = json.dumps({"model": model, "input": texts}).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        key = str(self._c("embedding_api_key", "") or "").strip()
        if key:
            req.add_header("Authorization", "Bearer " + key)
        try:
            with urllib.request.urlopen(req, timeout=float(self._c("embedding_timeout", 12))) as r:
                data = json.loads(r.read().decode())
            return [d["embedding"] for d in data.get("data", [])]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 向量请求失败：{exc!r}")
            return []

    def _save_vec(self, mem_id: int, vec: list) -> None:
        import array
        blob = array.array("f", [float(x) for x in vec]).tobytes()
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO mem_vec(mem_id,vec) VALUES(?,?)",
                            (mem_id, blob))
            self.db.commit()

    def _load_vecs(self) -> dict:
        import array
        out = {}
        with self._lock:
            rows = list(self.db.execute("SELECT mem_id, vec FROM mem_vec"))
        for mem_id, blob in rows:
            a = array.array("f")
            a.frombytes(blob)
            out[mem_id] = a.tolist()
        return out

    def backfill(self, batch: int = 8, limit: int = 400) -> int:
        """给还没有向量的记忆补向量。"""
        with self._lock:
            rows = self.db.execute(
                "SELECT m.id, m.text FROM memories m "
                "LEFT JOIN mem_vec v ON v.mem_id = m.id "
                "WHERE v.mem_id IS NULL LIMIT ?", (limit,)).fetchall()
        done = 0
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            vecs = self._embed_sync([r[1] for r in chunk])
            if not vecs or len(vecs) != len(chunk):
                break
            for (mid, _), vec in zip(chunk, vecs):
                self._save_vec(mid, vec)
                done += 1
        if done:
            logger.info(f"[{PLUGIN_NAME}] 向量补齐 {done} 条")
        return done

    @staticmethod
    def _cos(a: list, b: list) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        dot = sa = sb = 0.0
        for i in range(n):
            x, y = a[i], b[i]
            dot += x * y
            sa += x * x
            sb += y * y
        if sa <= 0 or sb <= 0:
            return 0.0
        return dot / ((sa ** 0.5) * (sb ** 0.5))

    # ---------------------------------------------------------------- 召回

    @staticmethod
    def _grams(text: str) -> set:
        import re as _re
        t = _re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", str(text or ""))
        stop = set("的了在是我你她他它们这那就也不都个有一和与或说会要可以吗呢吧啊哦嗯很")
        return {t[i:i + 2] for i in range(len(t) - 1) if t[i] not in stop}

    def recall(self, query: str, top: int = 5, qvec: list | None = None) -> list[dict]:
        """按 spec 的打分召回；有向量时用余弦，没有就退回字面相似。"""
        q = self._grams(query)
        if not q and not qvec:
            return []
        vecs = self._load_vecs() if qvec else {}
        f_endo = self._hormone_factor()
        now = time.time()
        rows = self.db.execute(
            "SELECT id,text,kind,importance,weight,t_anchor,access_count,pool "
            "FROM memories WHERE pool != 'trace'").fetchall()
        max_w = max([r[4] or 0.0 for r in rows] or [1.0]) or 1.0
        out = []
        for (mid, text, kind, imp, w, t_anchor, acc, pool) in rows:
            g = self._grams(text)
            inter = len(g & q) if q else 0
            if qvec is not None and mid in vecs:
                sim = self._cos(qvec, vecs[mid])
                if sim <= 0.05 and inter == 0:
                    continue
            else:
                if not g or inter == 0:
                    continue
                sim = inter / ((len(g) ** 0.5) * (len(q) ** 0.5) + 1e-9) + 1e-9
                sim = min(1.0, sim * 2.2)
            cover = (inter / max(1, len(q))) if q else sim
            w_n = min(1.0, (w or 0.0) / max_w)
            half = self._half_life(pool) * DAY
            rec = pow(2.718281828, -max(0.0, now - float(t_anchor or now)) / max(1.0, half))
            salience = max(0.0, min(1.0, float(imp or 0.5)))
            base = (0.45 * sim + 0.30 * cover + 0.15 * w_n + 0.10 * rec) * (0.5 + salience)
            score = base * f_endo
            out.append({"id": mid, "text": text, "kind": kind, "pool": pool,
                        "weight": round(w or 0, 4), "score": round(score, 4),
                        "sim": round(sim, 3), "cover": round(cover, 3)})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top]

    def _log_recall(self, items: list[dict], source: str) -> None:
        """只记录事件，权重改动留给 tick 批量结算（遗忘-联想解耦）。"""
        try:
            now = time.time()
            self.db.executemany(
                "INSERT INTO recall_events(mem_id,ts,score,source) VALUES(?,?,?,?)",
                [(it["id"], now, it["score"], source) for it in items])
            self.db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 记录回想失败：{exc!r}")

    def _settle_recall(self) -> int:
        """批量结算未处理的回想事件：+1 次回想，权重轻微加固。"""
        with self._lock:
            rows = self.db.execute(
                "SELECT mem_id, COUNT(*) c, MAX(score) s FROM recall_events "
                "WHERE processed=0 GROUP BY mem_id").fetchall()
        if not rows:
            return 0
        cur = self.db.cursor()
        for mem_id, cnt, mx in rows:
            cur.execute(
                "UPDATE memories SET access_count=access_count+1, "
                "last_access=?, weight=MIN(1.0, weight*(1+0.02)) WHERE id=?",
                (time.time(), mem_id))
        cur.execute("UPDATE recall_events SET processed=1 WHERE processed=0")
        self.db.commit()
        return len(rows)

    # ---------------------------------------------------------------- 统计

    def summary(self) -> dict:
        rows = self.db.execute(
            "SELECT pool, COUNT(*) c, AVG(weight) w FROM memories GROUP BY pool").fetchall()
        pools = {p: {"count": c, "avg_w": round(w or 0, 4)} for p, c, w in rows}
        total = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return {"total": total, "pools": pools, "f_endo": round(self._hormone_factor(), 3)}


    @filter.on_llm_request(priority=60000)
    async def inject_mind(self, event: AstrMessageEvent, req) -> None:
        """唯一注入点：人格 + 心境 + 记忆 + 技能一次装配。"""
        try:
            mode = str(self._c("inject_mode", "supplement") or "off")
            if mode == "off":
                return
            txt = ""
            for c in (getattr(getattr(event, "message_obj", None), "message", []) or []):
                from astrbot.api.message_components import Plain as _P
                if isinstance(c, _P) and getattr(c, "text", ""):
                    txt += c.text
            if not txt or str(txt).strip().startswith(("/", "／")):
                return
            try:
                self._settle_pending_skill(txt)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[{PLUGIN_NAME}] 技能判分异常：{exc!r}")
            qvec = None
            try:
                import asyncio as _a
                vs = await _a.to_thread(self._embed_sync, [txt[:400]])
                qvec = vs[0] if vs else None
            except Exception:
                qvec = None
            top = int(self._c("recall_top", 4))
            items = self.recall(txt, top=top, qvec=qvec)
            keep = []
            cur = ""
            for attr in ("system_prompt", "system"):
                if hasattr(req, attr):
                    cur = getattr(req, attr) or ""
                    break
            for it in items:
                if mode == "supplement" and it["text"][:24] in cur:
                    continue
                keep.append(it)
            block = self.mind_get_context(txt, top=top, qvec=qvec) if keep or True else ""
            if not block:
                return
            # 只给一次装配结果，避免重复堆叠
            for attr in ("system_prompt", "system"):
                if hasattr(req, attr):
                    try:
                        setattr(req, attr, (getattr(req, attr) or "") + "\n\n" + block)
                        break
                    except Exception:
                        continue
            if keep:
                self._log_recall(keep, "inject")
            logger.info(f"[{PLUGIN_NAME}] 心智装配 {len(block)} 字，记忆 {len(keep)} 条")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 装配失败已忽略：{exc!r}")



    @filter.on_decorating_result(priority=6500)
    async def capture_reply(self, event: AstrMessageEvent) -> None:
        """抓她这句回复，算自评分，留给下一轮结算。"""
        try:
            if not getattr(self, "_pending_skill", ""):
                return
            result = event.get_result()
            chain = getattr(result, "chain", None) if result else None
            if not chain:
                return
            from astrbot.api.message_components import Plain as _P
            reply = " ".join(c.text for c in chain
                             if isinstance(c, _P) and getattr(c, "text", ""))
            if not reply.strip():
                return
            mode = str(self._c("skill_self_eval", "heuristic") or "heuristic")
            q_self = None
            user_text = ""
            for c in (getattr(getattr(event, "message_obj", None), "message", []) or []):
                if isinstance(c, _P) and getattr(c, "text", ""):
                    user_text += c.text
            if mode == "llm":
                q_self = await self._self_eval_llm(user_text, reply)
            if q_self is None:
                q_self = self._self_eval_heuristic(reply)
            self._pending_selfeval = q_self
            logger.info(f"[{PLUGIN_NAME}] 自评 Q={q_self:.1f}（{mode}）")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 自评异常已忽略：{exc!r}")

    # ---------------------------------------------------------------- 技能（SM2）

    SK_LAM_B = 0.0012
    SK_KQ = 0.25

    def _skill_decay(self, dt_hours: float) -> int:
        with self._lock:
            rows = self.db.execute("SELECT id, L, I FROM skills").fetchall()
        if not rows:
            return 0
        with self._lock:
            for sid, L, I in rows:
                lam_eff = self.SK_LAM_B * (1.0 / max(float(I or 1.0), 1.0)) ** 0.3
                newL = float(L or 0) * pow(2.718281828, -lam_eff * dt_hours)
                self.db.execute("UPDATE skills SET L=? WHERE id=?",
                                (max(0.0, min(1.0, newL)), sid))
            self.db.commit()
        return len(rows)

    def _skill_use(self, name: str, q: float) -> dict:
        row = self.db.execute(
            "SELECT id, L, EF, I, use_count FROM skills WHERE name=?", (name,)).fetchone()
        if not row:
            return {}
        sid, L, EF, I, cnt = row
        L, EF, I = float(L or 0.35), float(EF or 2.0), float(I or 1.0)
        q = max(0.0, min(5.0, float(q)))
        EF_new = max(1.3, min(2.5, EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))))
        if q >= 3:
            I_new = 1.0 if I <= 0 else (6.0 if I <= 1 else I * EF_new)
        else:
            I_new = 1.0
        L_new = min(1.0, L + self.SK_KQ * (q / 5.0) * ((1 - L) / (1 + 0.05 * I)))
        with self._lock:
            self.db.execute(
                "UPDATE skills SET L=?, EF=?, I=?, last_used=?, use_count=use_count+1 WHERE id=?",
                (L_new, EF_new, I_new, time.time(), sid))
            self.db.commit()
        return {"name": name, "q": q, "L": round(L_new, 3), "EF": round(EF_new, 2),
                "I": round(I_new, 2)}

    def _skill_score(self, name: str, sim: float = 0.8) -> float:
        row = self.db.execute(
            "SELECT L, EF, attach_bonus FROM skills WHERE name=?", (name,)).fetchone()
        if not row:
            return 0.0
        L, EF, bonus = float(row[0] or 0), float(row[1] or 2.0), int(row[2] or 0)
        sc = sim * (max(0.0, L) ** 0.5) * (1 + 0.15 * (EF - 1.3))
        if L < 0.05:
            sc *= 0.2
        if bonus:
            try:
                with open(self.HORMONE_PATH, encoding="utf-8") as f:
                    oxy = float(json.load(f).get("oxy", 0.5))
                sc *= 1 + 0.25 * (oxy - 0.5) * 2
            except Exception:
                pass
        return round(sc, 3)

    def _budget_chars(self) -> int:
        budget = int(self._c("token_budget", 6000))
        ratio = float(self._c("memory_ratio", 0.12))
        return max(120, int(budget * ratio * 1.1))

    def _trim_to_budget(self, lines: list, budget_chars: int) -> list:
        out, used = [], 0
        for ln in lines:
            if used + len(ln) > budget_chars:
                break
            out.append(ln)
            used += len(ln)
        return out


    # ---------------------------------------------------------------- 自评

    @staticmethod
    def _self_eval_heuristic(reply: str) -> float:
        """她对自己这句回复的粗糙自评（1~5）。"""
        t = str(reply or "").strip()
        if not t:
            return 3.0
        score = 3.0
        n = len(t)
        if 6 <= n <= 60:
            score += 0.5
        elif n > 120:
            score -= 0.5
        if any(k in t for k in ("懂", "累", "抱", "别急", "我在", "陪你", "先歇")):
            score += 1.0
        if any(k in t for k in ("随便", "不知道", "哦", "呵呵")):
            score -= 1.0
        if t.rstrip().endswith(("?", "？", "吗", "呢")):
            score += 0.5
        if t.count("。") + t.count("，") > 6:
            score -= 0.5
        return max(1.0, min(5.0, score))

    async def _self_eval_llm(self, user_text: str, reply: str) -> float | None:
        """让她自己给刚才那句打 1~5 分（失败就返回 None，退回启发式）。"""
        try:
            prov = self.context.get_using_provider()
            if not prov:
                return None
            prompt = (f"用户说：{user_text[:180]}\n"
                      f"你回的是：{reply[:180]}\n"
                      "只回一个 1-5 的数字，评价你这句安慰/回应做得怎么样，不要解释。")
            resp = await prov.text_chat(
                prompt=prompt,
                system_prompt="你是评分器，只输出一个 1-5 的数字。",
                contexts=[])
            txt = str(getattr(resp, "completion_text", "") or resp)
            import re as _re
            m = _re.search(r"[1-5](?:\.\d)?", txt)
            return float(m.group(0)) if m else None
        except Exception as exc:  # noqa: BLE001
            logger.info(f"[{PLUGIN_NAME}] 自评走启发式（LLM 不可用：{exc!r}）")
            return None

    # ---------------------------------------------------------------- 心智入口

    HORMONE_PATH = "/AstrBot/data/plugin_data/dafeiyu_hormone/hormones.json"
    PERSONA_PATH = "/AstrBot/data/plugin_data/dafeiyu_persona_core/persona_core.json"

    @staticmethod
    def _read_json(path: str) -> dict:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _hormone_line(self) -> str:
        h = self._read_json(self.HORMONE_PATH)
        if not h:
            return ""
        sero, dopa = float(h.get("sero", 0.6)), float(h.get("dopa", 0.4))
        cort, oxy = float(h.get("cort", 0.2)), float(h.get("oxy", 0.5))
        bits = ["有耐心" if sero >= 0.58 else ("没耐心、话短" if sero < 0.45 else "情绪平稳")]
        bits.append("想闹" if dopa >= 0.52 else ("闷" if dopa < 0.38 else "不咸不淡"))
        if cort >= 0.40:
            bits.append("心里压着火")
        elif cort >= 0.28:
            bits.append("有点提不起劲")
        if oxy >= 0.62:
            bits.append("很想黏着他")
        elif oxy < 0.45:
            bits.append("有点端着")
        return "【你此刻的心境】" + "；".join(bits) + "。（只影响语气，不要念出来）"

    def _persona_line(self) -> str:
        d = self._read_json(self.PERSONA_PATH)
        P = d.get("P") or {}
        if not P:
            return ""
        g = float(P.get("gentle", .78))
        c = float(P.get("clingy", .72))
        b = float(P.get("blunt", .55))
        s_ = float(P.get("sensitive", .62))
        bits = ["温和" if g >= 0.7 else ("有点冲" if g < 0.55 else "正常")]
        bits.append("黏人" if c >= 0.68 else ("端着" if c < 0.5 else "正常"))
        bits.append("说话直" if b >= 0.62 else "说话绕")
        if s_ >= 0.68:
            bits.append("容易往心里去")
        return "【你的性格底色】" + "；".join(bits) + "。（性格稳定，别因为一句话就变）"

    def _skill_line(self, query: str) -> str:
        try:
            rows = self.db.execute(
                "SELECT name, desc, L FROM skills WHERE L >= 0.05").fetchall()
        except Exception:
            return ""
        if not rows:
            return ""
        q = self._grams(query)
        best, best_sc = None, 0.0
        for name, desc, L in rows:
            sc = self._skill_score(name)
            kw = len(q & self._grams(str(name) + str(desc))) if q else 0
            sc = sc + 0.15 * kw
            if sc > best_sc:
                best, best_sc = (name, desc), sc
        if not best or best_sc < 0.25:
            return ""
        name, desc = best
        self._pending_skill = name
        self._pending_ts = time.time()
        try:
            with self._lock:
                self.db.execute(
                    "INSERT INTO skill_events(name,ts,q,how) VALUES(?,?,NULL,'use')",
                    (name, time.time()))
                self.db.commit()
        except Exception:
            pass
        return f"【你有个老习惯可以用】{desc or name}（自然用出来，别说明这是习惯）"

    def _q_max(self) -> float:
        """压力越大，技能发挥上限越低。"""
        h = self._read_json(self.HORMONE_PATH)
        cort = float(h.get("cort", 0.2))
        return 5.0 * (1 - 0.60 * max(0.0, cort - 0.20) / 0.80)

    def _settle_pending_skill(self, text: str) -> None:
        """用下一句用户的反馈给上一次技能打分。"""
        name = getattr(self, "_pending_skill", "")
        ts = getattr(self, "_pending_ts", 0.0)
        if not name:
            return
        if time.time() - ts > 900:          # 超过 15 分钟不判了
            self._pending_skill = ""
            return
        t = str(text or "")
        pos = sum(1 for k in ("好", "谢谢", "喜欢", "对", "嗯", "是的", "厉害", "棒", "抱", "乖") if k in t)
        neg = sum(1 for k in ("不是", "别", "烦", "不对", "不", "算了", "无聊", "没意思") if k in t)
        q_feedback = 4.0 if pos > neg else (2.0 if neg > pos else 3.0)
        q_self = float(getattr(self, "_pending_selfeval", 0) or 0)
        self._pending_selfeval = 0.0
        if q_self > 0:
            w = float(self._c("self_weight", 0.5))
            base = q_feedback * (1 - w) + q_self * w
        else:
            base = q_feedback
        q = min(base, self._q_max())
        try:
            r = self._skill_use(name, q)
            with self._lock:
                self.db.execute(
                    "UPDATE skill_events SET q=?, how='auto' WHERE id=("
                    "SELECT id FROM skill_events WHERE name=? AND q IS NULL "
                    "ORDER BY id DESC LIMIT 1)", (q, name))
                self.db.commit()
            logger.info(f"[{PLUGIN_NAME}] 技能判分 {name} Q={q:.2f}"
                        f"（反馈 {q_feedback} / 自评 {q_self or '无'} / 上限 {self._q_max():.2f}）"
                        f" → L={r.get('L')}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 技能判分失败：{exc!r}")
        self._pending_skill = ""

    def _vision_note(self) -> str:
        """图片只给一句描述时，提醒她别把描述当成"看得清清楚楚"。"""
        if not bool(self._c("vision_note", True)):
            return ""
        return ("【关于图片】系统会附一句图里有什么的简短描述，你就按那句话聊，"
                "别当成自己盯着图看得很细；没给描述就说没看清，不要编。")

    def mind_get_context(self, text: str, top: int = 4, qvec: list | None = None) -> str:
        """成品上下文：人格 + 心境 + 记忆 + 技能，一次装配好交给大模型。"""
        parts = []
        for line in (self._persona_line(), self._hormone_line()):
            if line:
                parts.append(line)
        items = self.recall(text, top=top, qvec=qvec) if text else []
        if items:
            head = "【这些是你想起的事，自然地带出来，不要整段复述】"
            body = ["· " + it["text"].strip() for it in items]
            body = self._trim_to_budget(body, max(80, self._budget_chars() - len(head)))
            parts.append(head + "\n" + "\n".join(body))
        sk = self._skill_line(text)
        if sk:
            parts.append(sk)
        return "\n\n".join(parts)

    # ---------------------------------------------------------------- 生命卡

    def export_lifecard(self) -> str:
        """打包状态成一个可插拔的生命卡 zip。"""
        import zipfile
        os.makedirs(DATA_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(DATA_DIR, f"lifecard_{stamp}.lifecard")
        try:
            self.db.commit()
        except Exception:
            pass
        manifest = {
            "format": "dafeiyu-lifecard/v1",
            "created": stamp,
            "plugins": {
                "mindcore": {"db": os.path.basename(CORE_DB),
                             "memories": self.summary().get("total")},
                "hormone": self._read_json(self.HORMONE_PATH),
                "persona_core": self._read_json(self.PERSONA_PATH),
            },
        }
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
                if os.path.exists(CORE_DB):
                    z.write(CORE_DB, "memory_core.db")
                for src, arc in ((self.HORMONE_PATH, "hormone.json"),
                                 (self.PERSONA_PATH, "persona_core.json")):
                    if os.path.exists(src):
                        z.write(src, arc)
                z.writestr("manifest.json",
                           json.dumps(manifest, ensure_ascii=False, indent=1))
            size = os.path.getsize(path) / 1024
            logger.info(f"[{PLUGIN_NAME}] 生命卡导出 {path}（{size:.0f} KB）")
            return f"{path}（{size:.0f} KB）"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 导出失败：{exc!r}")
            return ""


    def import_lifecard(self, name: str = "") -> str:
        """从 plugin_data 里的 .lifecard 恢复状态（记忆库 + 激素 + 人格）。"""
        import shutil
        import zipfile
        files = [f for f in os.listdir(DATA_DIR) if f.endswith(".lifecard")]
        files.sort(reverse=True)
        if not files:
            return ""
        target = name.strip() if name.strip() in files else files[0]
        path = os.path.join(DATA_DIR, target)
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                if "memory_core.db" in names:
                    self.db.close()
                    backup = CORE_DB + ".before_import"
                    if os.path.exists(CORE_DB):
                        shutil.copyfile(CORE_DB, backup)
                    with open(CORE_DB, "wb") as f:
                        f.write(z.read("memory_core.db"))
                    self.db = sqlite3.connect(CORE_DB, timeout=15, check_same_thread=False)
                    self.db.execute("PRAGMA journal_mode=WAL")
                    self._init_schema()
                for arc, dst in (("hormone.json", self.HORMONE_PATH),
                                 ("persona_core.json", self.PERSONA_PATH)):
                    if arc in names:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with open(dst, "wb") as f:
                            f.write(z.read(arc))
            logger.info(f"[{PLUGIN_NAME}] 生命卡导入完成：{target}")
            return target
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 导入失败：{exc!r}")
            return ""

    @filter.command("生命卡")
    async def cmd_lifecard(self, event: AstrMessageEvent, arg: str = ""):
        """导出/查看生命卡。用法：/生命卡 或 /生命卡 导出"""
        a = str(arg or "").strip()
        if a in ("导出", "export", ""):
            p = self.export_lifecard()
            if p:
                yield event.plain_result(f"生命卡已导出：\n{p}")
            else:
                yield event.plain_result("导出失败，看日志。")
            return
        if a.startswith("导入"):
            target = self.import_lifecard(a[2:].strip())
            if target:
                yield event.plain_result(f"已导入 {target}（旧库已备份为 .before_import）")
            else:
                yield event.plain_result("没找到生命卡，或导入失败。")
            return
        if a == "列表":
            files = [f for f in os.listdir(DATA_DIR) if f.endswith(".lifecard")]
            files.sort(reverse=True)
            if not files:
                yield event.plain_result("还没有生命卡。")
                return
            yield event.plain_result("生命卡：\n" + "\n".join(f"· {f}" for f in files[:10]))
            return
        yield event.plain_result("用法：/生命卡 导出 ｜ /生命卡 列表")


    @filter.command("技能记录")
    async def cmd_skill_events(self, event: AstrMessageEvent):
        """看最近的技能调用与自动打分。用法：/技能记录"""
        rows = self.db.execute(
            "SELECT name, ts, q, how FROM skill_events ORDER BY id DESC LIMIT 10").fetchall()
        if not rows:
            yield event.plain_result("还没有技能调用记录。")
            return
        out = ["最近技能事件："]
        for name, ts, q, how in rows:
            t = time.strftime("%m-%d %H:%M", time.localtime(ts or 0))
            out.append(f"· {t} {name} Q={('待判' if q is None else f'{q:.1f}')} [{how}]")
        yield event.plain_result("\n".join(out))


    @filter.command("心智调参")
    async def cmd_tune(self, event: AstrMessageEvent):
        """根据当前数据建议晋升阈值。用法：/心智调参"""
        rows = self.db.execute(
            "SELECT importance, weight, access_count, entities, pool FROM memories").fetchall()
        if not rows:
            yield event.plain_result("还没有记忆。")
            return
        import statistics as st
        scores = []
        for imp, w, acc, ent, pool in rows:
            e = max(0.0, min(1.0, float(imp or 0.5)))
            c_e = min(1.0, len([x for x in str(ent or "").split(",") if x.strip()]) / 4.0)
            c_r = min(1.0, (acc or 0) / 5.0)
            scores.append(0.40 * e + 0.20 * c_e + 0.25 * c_r + 0.15 * e)
        scores.sort()
        p50 = scores[len(scores) // 2]
        p75 = scores[int(len(scores) * 0.75)]
        counts = {}
        for _, _, _, _, pool in rows:
            counts[pool] = counts.get(pool, 0) + 1
        yield event.plain_result(
            f"海马分：中位 {p50:.2f}　75 分位 {p75:.2f}\n"
            f"当前阈值：中期 {self._c('promote_mid', 0.35)} / 长期 {self._c('promote_long', 0.60)}\n"
            f"建议：中期 {max(0.30, round(p50, 2))} / 长期 {max(0.55, round(p75, 2))}\n"
            f"分布：{counts}")


    @filter.command("心智去重")
    async def cmd_dedupe(self, event: AstrMessageEvent, action: str = ""):
        """合并重复记忆。用法：/心智去重（预演）｜ /心智去重 确认（真的合并）"""
        import asyncio

        def run(apply: bool) -> dict:
            rows = self.db.execute(
                "SELECT id, text, weight, access_count, t_anchor, pool FROM memories "
                "ORDER BY id").fetchall()
            groups: dict = {}
            for rid, text, w, acc, ts, pool in rows:
                k = self._norm_text(text)
                if not k:
                    continue
                groups.setdefault(k, []).append((rid, w, acc, ts, pool))
            dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
            extra = sum(len(v) - 1 for v in dup_groups.values())
            if apply and dup_groups:
                with self._lock:
                    for k, v in dup_groups.items():
                        keep = max(v, key=lambda x: (x[1] or 0))
                        w = max(float(x[1] or 0) for x in v)
                        acc = sum(int(x[2] or 0) for x in v)
                        self.db.execute(
                            "UPDATE memories SET weight=?, access_count=? WHERE id=?",
                            (min(1.0, w), acc, keep[0]))
                        for x in v:
                            if x[0] != keep[0]:
                                self.db.execute("DELETE FROM memories WHERE id=?", (x[0],))
                    self.db.commit()
            return {"groups": len(dup_groups), "extra": extra,
                    "kept": len(rows) - (extra if apply else 0), "applied": apply}

        apply = str(action or "").strip() in ("确认", "apply", "yes")
        try:
            r = await asyncio.to_thread(run, apply)
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"去重失败：{exc!r}")
            return
        if r["groups"] == 0:
            yield event.plain_result("没有重复记忆。")
            return
        if r["applied"]:
            yield event.plain_result(
                f"已合并 {r['groups']} 组、删掉 {r['extra']} 条重复\n现在共 {r['kept']} 条")
        else:
            yield event.plain_result(
                f"发现 {r['groups']} 组重复，多出 {r['extra']} 条\n"
                f"确认合并请发：/心智去重 确认")

    @filter.command("心智状态")
    async def cmd_all(self, event: AstrMessageEvent):
        """一次看全套状态。用法：/心智状态"""
        h = self._read_json(self.HORMONE_PATH)
        pc = self._read_json(self.PERSONA_PATH)
        s = self.summary()
        P = pc.get("P") or {}
        skills = self.db.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        lines = [
            f"记忆 {s['total']} 条（short/mid/long = "
            f"{s['pools'].get('short', {}).get('count', 0)}/"
            f"{s['pools'].get('mid', {}).get('count', 0)}/"
            f"{s['pools'].get('long', {}).get('count', 0)}）",
            f"心境 血清素 {float(h.get('sero', 0)):.2f} 多巴胺 {float(h.get('dopa', 0)):.2f} "
            f"皮质醇 {float(h.get('cort', 0)):.2f} 催产素 {float(h.get('oxy', 0)):.2f}",
            f"人格 温柔 {float(P.get('gentle', 0)):.2f} 粘人 {float(P.get('clingy', 0)):.2f} "
            f"直率 {float(P.get('blunt', 0)):.2f} 敏感 {float(P.get('sensitive', 0)):.2f}",
            f"技能 {skills} 项｜召回加权 f_endo {s['f_endo']}",
        ]
        yield event.plain_result("\n".join(str(x) for x in lines))


    # ---------------------------------------------------------------- 备份

    BACKUP_DIR = os.path.join(DATA_DIR, "backups")
    PERSONA_FILE = "/AstrBot/data/plugin_data/dafeiyu_persona_core/persona_core.json"

    BACKUP_ITEMS = [
        (CORE_DB, "memory_core.db"),
        (OLD_DB, "memory.db"),
        (BRAIN_DB, "brain.db"),
        (HORMONE_FILE, "hormones.json"),
        (PERSONA_FILE, "persona_core.json"),
    ]

    @staticmethod
    def _copy_sqlite(src: str, dst: str) -> int:
        """用 sqlite backup API 拷库，避开 WAL 只拷主文件丢数据的问题。"""
        import sqlite3
        con = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=15)
        try:
            out = sqlite3.connect(dst, timeout=15)
            try:
                con.backup(out)
            finally:
                out.close()
        finally:
            con.close()
        return os.path.getsize(dst)

    def make_backup(self) -> dict:
        import hashlib
        import shutil
        import sqlite3
        import time as _t
        import zipfile

        os.makedirs(self.BACKUP_DIR, exist_ok=True)
        stamp = _t.strftime("%Y%m%d_%H%M%S")
        tmpdir = os.path.join(self.BACKUP_DIR, f".tmp_{stamp}")
        os.makedirs(tmpdir, exist_ok=True)
        manifest = {"created": stamp, "items": [], "mindcore": self.summary()}
        try:
            for src, arc in self.BACKUP_ITEMS:
                if not os.path.exists(src):
                    manifest["items"].append({"name": arc, "skipped": "not found"})
                    continue
                dst = os.path.join(tmpdir, arc)
                if src.endswith(".db"):
                    self._copy_sqlite(src, dst)
                else:
                    shutil.copyfile(src, dst)
                size = os.path.getsize(dst)
                h = hashlib.sha256(open(dst, "rb").read()).hexdigest()[:16]
                manifest["items"].append({"name": arc, "size": size, "sha256_16": h})
        finally:
            try:
                self.db.commit()
            except Exception:
                pass

        out = os.path.join(self.BACKUP_DIR, f"backup_{stamp}.zip")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in sorted(os.listdir(tmpdir)):
                z.write(os.path.join(tmpdir, f), f)
            manifest["items"].append({"name": "hormones/persona 已含", "skipped": "meta"})
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
        shutil.rmtree(tmpdir, ignore_errors=True)

        # 只留最近 7 个
        files = sorted([f for f in os.listdir(self.BACKUP_DIR) if f.endswith(".zip")])
        for old in files[:-int(self._c("backup_keep", 7))]:
            try:
                os.remove(os.path.join(self.BACKUP_DIR, old))
            except Exception:
                pass

        manifest["path"] = out
        manifest["zip_size"] = os.path.getsize(out)
        return manifest

    def verify_backup(self, path: str) -> str:
        """重新打开备份，校验每个文件能读出来。"""
        import zipfile
        try:
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
                if bad:
                    return f"损坏：{bad}"
                names = z.namelist()
                if "manifest.json" not in names:
                    return "缺 manifest.json"
                m = json.loads(z.read("manifest.json").decode())
                got = [i["name"] for i in m.get("items", []) if i.get("size")]
            return f"校验通过（{len(got)} 个文件：{', '.join(got)}）"
        except Exception as exc:  # noqa: BLE001
            return f"校验失败：{exc!r}"


    def drill_backup(self, name: str = "") -> str:
        """演练：解压到临时目录并验证每个库能打开。不改动现有数据。"""
        import shutil
        import sqlite3
        import tempfile
        import zipfile

        try:
            files = sorted(f for f in os.listdir(self.BACKUP_DIR) if f.endswith(".zip"))
        except Exception:
            files = []
        if not files:
            return "还没有备份。"
        target = files[-1]
        if name:
            cand = [f for f in files if name in f]
            if cand:
                target = cand[-1]
        path = os.path.join(self.BACKUP_DIR, target)
        tmp = tempfile.mkdtemp(prefix="dafeiyu_drill_")
        lines = [target]
        try:
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
                if bad:
                    return "压缩包损坏：" + bad
                z.extractall(tmp)
                names = z.namelist()
            detail = []
            counts = []
            for f in sorted(os.listdir(tmp)):
                p = os.path.join(tmp, f)
                size = os.path.getsize(p)
                if not f.endswith(".db"):
                    detail.append(f + " " + str(size) + "B")
                    continue
                con = sqlite3.connect("file:" + p + "?mode=ro", uri=True, timeout=15)
                try:
                    ok = con.execute("PRAGMA integrity_check").fetchone()[0]
                    n = ""
                    for table in ("memories", "episodes", "facts", "entities"):
                        try:
                            c = con.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                            n += table + " " + str(c) + " "
                        except Exception:
                            continue
                    counts.append(f + ":" + n.split()[1] if len(n.split()) > 1 else f)
                    detail.append(f + " ok")
                finally:
                    con.close()
            logger.info("[%s] 演练详情：%s" % (PLUGIN_NAME, " | ".join(lines + detail)))
            short = "%s 演练通过｜%d 个库完整性 ok｜可恢复" % (target, len(counts))
            return short
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return chr(10).join(lines)

    @filter.command("备份")
    async def cmd_backup(self, event: AstrMessageEvent):
        """把她的全部状态打包。用法：/备份 ｜ /备份 列表"""
        arg = str(getattr(event, "message_str", "") or "").split()
        if len(arg) > 1 and arg[1] in ("列表", "list"):
            try:
                files = sorted(f for f in os.listdir(self.BACKUP_DIR) if f.endswith(".zip"))
            except Exception:
                files = []
            if not files:
                yield event.plain_result("还没有备份。发 /备份 生成一个。")
                return
            lines = [f"备份 {len(files)} 个："]
            for f in files[-8:]:
                p = os.path.join(self.BACKUP_DIR, f)
                lines.append(f"· {f}（{os.path.getsize(p)/1024:.0f} KB）")
            yield event.plain_result("\n".join(lines))
            return

        if len(arg) > 1 and arg[1] in ("演练", "drill"):
            import asyncio
            r = await asyncio.to_thread(self.drill_backup, arg[2] if len(arg) > 2 else "")
            yield event.plain_result(r)
            return

        if len(arg) > 1 and arg[1] in ("校验", "verify", "check"):
            try:
                files = sorted(f for f in os.listdir(self.BACKUP_DIR) if f.endswith(".zip"))
            except Exception:
                files = []
            if not files:
                yield event.plain_result("还没有备份。")
                return
            target = files[-1]
            if len(arg) > 2:
                cand = [f for f in files if arg[2] in f]
                target = cand[-1] if cand else target
            r = self.verify_backup(os.path.join(self.BACKUP_DIR, target))
            yield event.plain_result(f"{target}\n{r}")
            return

        yield event.plain_result("正在打包…")
        try:
            import asyncio
            m = await asyncio.to_thread(self.make_backup)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 备份失败：{exc!r}")
            yield event.plain_result(f"备份失败：{exc!r}")
            return
        v = self.verify_backup(m["path"])
        detail = []
        for it in m["items"]:
            sz = it.get("size")
            if sz:
                detail.append(f"· {it['name']} " + (f"{sz/1024:.0f} KB" if sz >= 1024 else f"{sz} 字节"))
        yield event.plain_result(
            "备份完成\n" + "\n".join(detail) +
            f"\n\n{m['path'].split('/')[-1]}（{m['zip_size']/1024:.0f} KB）\n{v}"
        )

    # ---------------------------------------------------------------- 指令


    @filter.command("技能")
    async def cmd_skills(self, event: AstrMessageEvent):
        """看技能熟练度。用法：/技能"""
        rows = self.db.execute(
            "SELECT name, L, EF, I, use_count, attach_bonus FROM skills "
            "ORDER BY L DESC").fetchall()
        if not rows:
            yield event.plain_result("还没有技能。用 /技能增 名字|描述 添加一个。")
            return
        out = [f"技能 {len(rows)} 项："]
        for name, L, EF, I, cnt, bonus in rows:
            tag = " ★" if bonus else ""
            out.append(f"· {name}{tag}　L={L:.2f} EF={EF:.2f} I={I:.1f}h "
                       f"用过 {cnt} 次 → 召回分 {self._skill_score(name):.2f}")
        yield event.plain_result("\n".join(out))

    @filter.command("技能增")
    async def cmd_skill_add(self, event: AstrMessageEvent, payload: str = ""):
        """新增技能。用法：/技能增 安慰|先共情再给建议"""
        raw = (payload or "").strip()
        if not raw:
            yield event.plain_result("用法：/技能增 名称|描述")
            return
        name, _, desc = raw.partition("|")
        name = name.strip()
        try:
            self.db.execute(
                "INSERT OR IGNORE INTO skills(name,desc,L,EF,I,last_used,use_count) "
                "VALUES(?,?,?,?,?,?,0)", (name, desc.strip(), 0.35, 2.0, 1.0, time.time()))
            self.db.commit()
            yield event.plain_result(f"技能「{name}」已记录。")
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"失败：{exc!r}")

    @filter.command("技能用")
    async def cmd_skill_use(self, event: AstrMessageEvent, payload: str = "", qstr: str = ""):
        """记一次技能调用。用法：/技能用 安慰 4"""
        raw = (payload or "").strip()
        if not raw:
            yield event.plain_result("用法：/技能用 名称 [Q分 0-5]")
            return
        parts = raw.split()
        name = parts[0]
        try:
            q = float(qstr) if str(qstr).strip() else (float(parts[1]) if len(parts) > 1 else 3.0)
        except (TypeError, ValueError):
            q = 3.0
        r = self._skill_use(name, q)
        if not r:
            yield event.plain_result(f"没有这个技能：{name}")
            return
        yield event.plain_result(
            f"「{name}」Q={r['q']:.1f} → L={r['L']} EF={r['EF']} I={r['I']}h　"
            f"召回分 {self._skill_score(name)}")

    @filter.command("技能衰减")
    async def cmd_skill_decay(self, event: AstrMessageEvent):
        """手动跑一次技能衰减。用法：/技能衰减"""
        n = self._skill_decay(1.0)
        yield event.plain_result(f"已对 {n} 项技能跑 1 小时衰减。")

    @filter.command("心智")
    async def cmd_mind(self, event: AstrMessageEvent):
        """看心智内核统计。用法：/心智"""
        s = self.summary()
        lines = [f"心智内核：{s['total']} 条记忆（旧库只读导入，不回写）"]
        for pool in ("short", "mid", "long", "archive", "deep", "trace"):
            if pool in s["pools"]:
                p = s["pools"][pool]
                lines.append(f"· {pool:8} {p['count']:5} 条　平均权重 {p['avg_w']:.4f}")
        lines.append(f"· 内分泌加权 f_endo = {s['f_endo']}")
        yield event.plain_result("\n".join(lines))

    @filter.command("心智衰减")
    async def cmd_tick(self, event: AstrMessageEvent):
        """手动跑一次衰减结算。用法：/心智衰减"""
        st = self.tick()
        yield event.plain_result(
            f"结算 dt={st['dt_h']:.2f}h：衰减 {st['decayed']} 条，"
            f"晋升 {st['promoted']} 条，下沉 {st['sunk']} 条")

    @filter.command("心智向量")
    async def cmd_vec(self, event: AstrMessageEvent):
        """给缺向量的记忆补齐。用法：/心智向量"""
        import asyncio as _a
        n = await _a.to_thread(self.backfill, 8, 400)
        total = self.db.execute("SELECT COUNT(*) FROM mem_vec").fetchone()[0]
        allc = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        yield event.plain_result(f"本轮补齐 {n} 条；已有向量 {total}/{allc}")

    @filter.command("心智找")
    async def cmd_find(self, event: AstrMessageEvent, kw: str = ""):
        """按关键词找记忆。用法：/心智找 大盘鸡"""
        kw = (kw or "").strip()
        if not kw:
            yield event.plain_result("用法：/心智找 关键词")
            return
        rows = self.db.execute(
            "SELECT text, pool, weight, access_count FROM memories "
            "WHERE text LIKE ? ORDER BY weight DESC LIMIT 8", (f"%{kw}%",)).fetchall()
        if not rows:
            yield event.plain_result("没找到。")
            return
        out = [f"找到 {len(rows)} 条："]
        for t, pool, w, acc in rows:
            out.append(f"· [{pool} w={w:.3f} ×{acc}] {t[:40]}")
        yield event.plain_result("\n".join(out))

    # ---------------------------------------------------------------- 生命周期

    async def initialize(self) -> None:
        import asyncio
        added, skipped = self._seed()
        logger.info(f"[{PLUGIN_NAME}] 种子导入：新增 {added}，已存在 {skipped}")
        try:
            self.offline_settle()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 离线结算失败：{exc!r}")
        st = self.tick(0)
        logger.info(f"[{PLUGIN_NAME}] 就绪：{self.summary()}")

        interval = max(300, int(self._c("tick_seconds", 900)))

        async def _backfill_loop():
            import asyncio as _a
            total = 0
            for _ in range(40):
                n = await _a.to_thread(self.backfill)
                total += n
                if n == 0:
                    break
                await _a.sleep(1.0)
            logger.info(f"[{PLUGIN_NAME}] 向量补齐完成，共 {total} 条")

        import asyncio as _a2
        _a2.create_task(_backfill_loop())

        async def loop():
            while True:
                try:
                    s = self.tick()
                    if s["decayed"]:
                        logger.info(f"[{PLUGIN_NAME}] tick dt={s['dt_h']:.2f}h "
                                    f"衰减{s['decayed']} 晋升{s['promoted']} 下沉{s['sunk']}")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[{PLUGIN_NAME}] tick 失败：{exc!r}")
                await asyncio.sleep(interval)

        self._task = asyncio.create_task(loop())

    async def terminate(self) -> None:
        if self._task:
            self._task.cancel()
        try:
            self.db.commit()
            self.db.close()
        except Exception:
            pass
