import json
import time
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import text

try:
    from open_webui.models.token_usage import TokenUsage, TokenUsageCreate
    from open_webui.utils.auth import get_admin_user, get_verified_user
    from open_webui.internal.db import get_db
except ImportError:
    from apps.webui.models.token_usage import TokenUsage, TokenUsageCreate
    from utils.utils import get_admin_user, get_verified_user
    from apps.webui.internal.db import get_db

router = APIRouter()


def extract_nt_id(email: Optional[str], name: Optional[str], user_id: str) -> str:
    if email and "@" in email:
        return email.split("@")[0].upper()
    if name:
        return name.strip().upper()
    return user_id


@router.get("/filters")
async def get_usage_filters(user=Depends(get_verified_user)):
    return TokenUsage.get_filter_options()


@router.get("/summary")
async def get_usage_summary(
    period: str = Query("daily", regex="^(daily|monthly|yearly|custom)$"),
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    nt_id: Optional[str] = Query(None),
    model_id: Optional[str] = Query(None),
    user=Depends(get_verified_user),
):
    now = int(time.time())
    start_ts = None
    end_ts = None

    if start_date:
        dt_start = datetime.strptime(start_date, "%Y-%m-%d")
        start_ts = int(dt_start.replace(hour=0, minute=0, second=0).timestamp())
    else:
        if period == "daily":
            start_ts = now - (30 * 86400)
        elif period == "monthly":
            start_ts = now - (365 * 86400)
        elif period == "yearly":
            start_ts = now - (5 * 365 * 86400)

    if end_date:
        dt_end = datetime.strptime(end_date, "%Y-%m-%d")
        end_ts = int(dt_end.replace(hour=23, minute=59, second=59).timestamp())

    return TokenUsage.query_usage_by_chat(
        period=period,
        start_ts=start_ts,
        end_ts=end_ts,
        nt_id=nt_id,
        model_id=model_id,
    )


@router.post("/backfill")
async def backfill_from_existing_chats(user=Depends(get_admin_user)):
    inserted = 0
    with get_db() as db:
        users = {
            r.id: (r.email, r.name)
            for r in db.execute(text('SELECT id, email, name FROM "user"')).fetchall()
        }
        chats = db.execute(
            text("SELECT id, user_id, title, chat, created_at FROM chat")
        ).fetchall()

        for chat_row in chats:
            chat_id = chat_row.id
            chat_title = getattr(chat_row, "title", None) or "Untitled Chat"
            user_id = chat_row.user_id
            email, u_name = users.get(user_id, (None, None))
            nt_id = extract_nt_id(email, u_name, user_id)

            try:
                chat_json = (
                    json.loads(chat_row.chat)
                    if isinstance(chat_row.chat, str)
                    else chat_row.chat
                )
            except Exception:
                continue

            if not isinstance(chat_json, dict):
                continue

            history_msgs = chat_json.get("history", {}).get("messages", {})
            msg_list = (
                list(history_msgs.values())
                if isinstance(history_msgs, dict) and history_msgs
                else (chat_json.get("messages") or [])
            )

            last_user_prompt = ""
            for msg in msg_list:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role == "user":
                    last_user_prompt = str(msg.get("content") or "")
                    continue
                if role != "assistant":
                    continue

                # If history map stores parentId, look up exact parent user prompt
                parent_id = msg.get("parentId")
                if parent_id and isinstance(history_msgs, dict) and parent_id in history_msgs:
                    parent_msg = history_msgs[parent_id]
                    if isinstance(parent_msg, dict) and parent_msg.get("content"):
                        last_user_prompt = str(parent_msg.get("content"))

                msg_id = msg.get("id")
                model_id = msg.get("model") or "unknown-model"
                response_text = str(msg.get("content") or "")
                usage = msg.get("usage") or msg.get("info", {})

                prompt_tokens = int(
                    usage.get("prompt_tokens")
                    or usage.get("prompt_eval_count")
                    or (max(1, len(last_user_prompt) // 4) if last_user_prompt else 0)
                )
                completion_tokens = int(
                    usage.get("completion_tokens")
                    or usage.get("eval_count")
                    or (max(1, len(response_text) // 4) if response_text else 0)
                )
                total_tokens = int(
                    usage.get("total_tokens") or (prompt_tokens + completion_tokens)
                )

                msg_ts = int(msg.get("timestamp") or chat_row.created_at or time.time())
                res = TokenUsage.insert_log(
                    TokenUsageCreate(
                        user_id=user_id,
                        nt_id=nt_id,
                        user_name=u_name or nt_id,
                        model_id=model_id,
                        chat_id=chat_id,
                        chat_title=chat_title,
                        message_id=f"{chat_id}:{msg_id}" if msg_id else None,
                        prompt_text=last_user_prompt,
                        response_text=response_text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        created_at=msg_ts,
                    )
                )
                if res:
                    inserted += 1

    return {"status": "ok", "inserted_records": inserted}