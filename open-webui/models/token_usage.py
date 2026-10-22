import time
import uuid
from typing import Optional
from pydantic import BaseModel
from sqlalchemy import Column, String, BigInteger, Integer, Text, Index, func, and_, text

try:
    from open_webui.internal.db import Base, get_db
except ImportError:
    from apps.webui.internal.db import Base, get_db


class TokenUsageLog(Base):
    __tablename__ = "token_usage_log"

    id = Column(String, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    nt_id = Column(String, index=True, nullable=False)
    user_name = Column(String, nullable=True)
    model_id = Column(String, index=True, nullable=False)
    chat_id = Column(String, index=True, nullable=False)
    chat_title = Column(String, nullable=True)
    message_id = Column(String, unique=True, nullable=True)
    prompt_text = Column(Text, nullable=True)
    response_text = Column(Text, nullable=True)
    prompt_tokens = Column(Integer, default=0, nullable=False)
    completion_tokens = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    created_at = Column(BigInteger, index=True, nullable=False)

    __table_args__ = (
        Index("idx_usage_chat_nt_model_time", "chat_id", "nt_id", "model_id", "created_at"),
    )


class TokenUsageCreate(BaseModel):
    user_id: str
    nt_id: str
    user_name: Optional[str] = ""
    model_id: str
    chat_id: str
    chat_title: Optional[str] = "Untitled Chat"
    message_id: Optional[str] = None
    prompt_text: Optional[str] = ""
    response_text: Optional[str] = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    created_at: Optional[int] = None


class TokenUsageTable:
    def __init__(self):
        with get_db() as db:
            TokenUsageLog.__table__.create(bind=db.get_bind(), checkfirst=True)
            # Safe auto-migration if table was created from the earlier schema
            for col_name, col_type in [
                ("chat_title", "TEXT"),
                ("prompt_text", "TEXT"),
                ("response_text", "TEXT"),
            ]:
                try:
                    db.execute(
                        text(f"ALTER TABLE token_usage_log ADD COLUMN {col_name} {col_type}")
                    )
                    db.commit()
                except Exception:
                    db.rollback()

    def insert_log(self, form_data: TokenUsageCreate) -> Optional[dict]:
        with get_db() as db:
            if form_data.message_id:
                existing = (
                    db.query(TokenUsageLog)
                    .filter_by(message_id=form_data.message_id)
                    .first()
                )
                if existing:
                    return None

            now_ts = form_data.created_at or int(time.time())
            total = form_data.total_tokens or (
                form_data.prompt_tokens + form_data.completion_tokens
            )
            record = TokenUsageLog(
                id=str(uuid.uuid4()),
                user_id=form_data.user_id,
                nt_id=form_data.nt_id,
                user_name=form_data.user_name,
                model_id=form_data.model_id,
                chat_id=form_data.chat_id,
                chat_title=form_data.chat_title or "Untitled Chat",
                message_id=form_data.message_id,
                prompt_text=form_data.prompt_text or "",
                response_text=form_data.response_text or "",
                prompt_tokens=form_data.prompt_tokens,
                completion_tokens=form_data.completion_tokens,
                total_tokens=total,
                created_at=now_ts,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            return {"id": record.id, "chat_id": record.chat_id}

    def get_filter_options(self) -> dict:
        with get_db() as db:
            nt_ids = [
                r[0]
                for r in db.query(TokenUsageLog.nt_id)
                .distinct()
                .order_by(TokenUsageLog.nt_id)
                .all()
                if r[0]
            ]
            models = [
                r[0]
                for r in db.query(TokenUsageLog.model_id)
                .distinct()
                .order_by(TokenUsageLog.model_id)
                .all()
                if r[0]
            ]
            return {"nt_ids": nt_ids, "models": models}

    def query_usage_by_chat(
        self,
        period: str = "daily",
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        nt_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> dict:
        with get_db() as db:
            dialect = db.get_bind().dialect.name
            if period == "yearly":
                fmt_sqlite, fmt_pg = "%Y", "YYYY"
            elif period == "monthly":
                fmt_sqlite, fmt_pg = "%Y-%m", "YYYY-MM"
            else:
                fmt_sqlite, fmt_pg = "%Y-%m-%d", "YYYY-MM-DD"

            if dialect == "postgresql":
                bucket_expr = func.to_char(
                    func.to_timestamp(TokenUsageLog.created_at), fmt_pg
                )
            else:
                bucket_expr = func.strftime(
                    fmt_sqlite, TokenUsageLog.created_at, "unixepoch", "localtime"
                )

            filters = []
            if start_ts is not None:
                filters.append(TokenUsageLog.created_at >= start_ts)
            if end_ts is not None:
                filters.append(TokenUsageLog.created_at <= end_ts)
            if nt_id and nt_id != "ALL":
                filters.append(TokenUsageLog.nt_id.ilike(f"%{nt_id.strip()}%"))
            if model_id and model_id != "ALL":
                filters.append(TokenUsageLog.model_id == model_id)

            # Fetch all matching message logs ordered chronologically within each chat
            logs = (
                db.query(
                    bucket_expr.label("period_bucket"),
                    TokenUsageLog,
                )
                .filter(and_(*filters))
                .order_by(TokenUsageLog.created_at.asc())
                .all()
            )

            # Group messages by (period_bucket, chat_id, model_id)
            chats_map = {}
            for period_bucket, row in logs:
                group_key = f"{period_bucket}::{row.chat_id}::{row.model_id}"
                if group_key not in chats_map:
                    chats_map[group_key] = {
                        "period": period_bucket,
                        "chat_id": row.chat_id,
                        "chat_title": row.chat_title or row.chat_id,
                        "nt_id": row.nt_id,
                        "user_name": row.user_name or row.nt_id,
                        "model_id": row.model_id,
                        "request_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "last_active": row.created_at,
                        "messages": [],
                    }

                chat_entry = chats_map[group_key]
                chat_entry["request_count"] += 1
                chat_entry["prompt_tokens"] += int(row.prompt_tokens or 0)
                chat_entry["completion_tokens"] += int(row.completion_tokens or 0)
                chat_entry["total_tokens"] += int(row.total_tokens or 0)
                chat_entry["last_active"] = max(chat_entry["last_active"], row.created_at)
                if row.chat_title and row.chat_title != "Untitled Chat":
                    chat_entry["chat_title"] = row.chat_title

                chat_entry["messages"].append(
                    {
                        "id": row.id,
                        "message_id": row.message_id,
                        "prompt_text": row.prompt_text or "",
                        "response_text": row.response_text or "",
                        "prompt_tokens": int(row.prompt_tokens or 0),
                        "completion_tokens": int(row.completion_tokens or 0),
                        "total_tokens": int(row.total_tokens or 0),
                        "created_at": row.created_at,
                    }
                )

            records = sorted(
                chats_map.values(),
                key=lambda x: (x["period"], x["last_active"]),
                reverse=True,
            )

            totals = {
                "total_chats": len({r["chat_id"] for r in records}),
                "total_requests": sum(r["request_count"] for r in records),
                "prompt_tokens": sum(r["prompt_tokens"] for r in records),
                "completion_tokens": sum(r["completion_tokens"] for r in records),
                "total_tokens": sum(r["total_tokens"] for r in records),
                "unique_nt_ids": len({r["nt_id"] for r in records}),
            }

            return {"totals": totals, "records": records}


TokenUsage = TokenUsageTable()