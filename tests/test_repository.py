from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from dealdesk.crm import CRMRepository, NotFoundError, ValidationError
from dealdesk.db.models import ActivityType, DealStage, Source
from dealdesk.db.session import session_scope
from tests.conftest import NOW, fixed_clock

Factory = sessionmaker[Session]


def repo_call(factory: Factory, fn):  # type: ignore[no-untyped-def]
    with session_scope(factory) as session:
        return fn(CRMRepository(session, clock=fixed_clock))


def test_stale_deals_are_open_deals_without_recent_activity(session_factory: Factory) -> None:
    deals = repo_call(session_factory, lambda r: r.list_deals(stale_days=21))
    assert {d.name for d in deals} == {
        "Fleet CRM rollout",
        "Loyalty agent pilot",
        "Service desk automation",
    }
    assert all(not d.is_closed and d.days_since_last_activity >= 21 for d in deals)


def test_pipeline_summary_totals(session_factory: Factory) -> None:
    summary = repo_call(session_factory, lambda r: r.pipeline_summary())
    assert summary.open_count == 13
    assert summary.open_amount == 851_000
    assert summary.won_amount == 136_000
    assert summary.lost_amount == 33_000
    by_owner = {o.owner: o.open_amount for o in summary.by_owner}
    assert by_owner == {"Lena Fischer": 375_000, "Jonas Berg": 285_000, "Mia Schneider": 191_000}


def test_filter_by_stage_and_owner(session_factory: Factory) -> None:
    deals = repo_call(session_factory, lambda r: r.list_deals(stage=DealStage.CONTRACT_SENT))
    assert [d.name for d in deals] == ["Claims intake agent", "Partner portal migration"]
    lena = repo_call(session_factory, lambda r: r.list_deals(owner="lena", open_only=True))
    assert len(lena) == 5
    assert all(d.owner == "Lena Fischer" for d in lena)


def test_closing_soon(session_factory: Factory) -> None:
    deals = repo_call(session_factory, lambda r: r.list_deals(closing_within_days=14))
    assert {d.name for d in deals} == {"Claims intake agent", "Partner portal migration"}


def test_overdue_tasks(session_factory: Factory) -> None:
    tasks = repo_call(session_factory, lambda r: r.list_tasks(overdue_only=True))
    assert [t.title for t in tasks] == ["Follow up on IT security questionnaire"]
    assert tasks[0].is_overdue


def test_search_is_case_insensitive(session_factory: Factory) -> None:
    hits = repo_call(session_factory, lambda r: r.search_companies("SEEBLICK"))
    assert [c.name for c in hits] == ["Seeblick Versicherung AG"]


def test_missing_deal_raises(session_factory: Factory) -> None:
    with pytest.raises(NotFoundError):
        repo_call(session_factory, lambda r: r.get_deal(999))


def test_create_task_defaults_to_deal_owner(session_factory: Factory) -> None:
    due = NOW.date() + timedelta(days=2)
    task = repo_call(session_factory, lambda r: r.create_task(4, "Call Erik", due))
    assert task.owner == "Jonas Berg"
    assert task.source == Source.AGENT


def test_create_task_rejects_past_due_date(session_factory: Factory) -> None:
    with pytest.raises(ValidationError):
        repo_call(
            session_factory, lambda r: r.create_task(4, "Too late", NOW.date() - timedelta(1))
        )


def test_update_stage_writes_audit_note(session_factory: Factory) -> None:
    row = repo_call(session_factory, lambda r: r.update_deal_stage(8, DealStage.CLOSED_WON))
    assert row.stage == DealStage.CLOSED_WON
    assert row.is_closed
    detail = repo_call(session_factory, lambda r: r.get_deal(8))
    assert "Stage changed from 'Contract sent' to 'Closed won'" in detail.activities[0].summary
    with pytest.raises(ValidationError):
        repo_call(session_factory, lambda r: r.update_deal_stage(8, DealStage.CLOSED_WON))


def test_log_activity_rejects_contact_from_other_company(session_factory: Factory) -> None:
    with pytest.raises(ValidationError):
        # contact 1 belongs to Nordlicht, deal 4 to Fjordway
        repo_call(
            session_factory, lambda r: r.log_activity(4, ActivityType.CALL, "x", contact_id=1)
        )


def test_log_activity_resets_staleness(session_factory: Factory) -> None:
    repo_call(session_factory, lambda r: r.log_activity(4, ActivityType.CALL, "Called Erik"))
    stale = repo_call(session_factory, lambda r: r.list_deals(stale_days=21))
    assert "Fleet CRM rollout" not in {d.name for d in stale}
