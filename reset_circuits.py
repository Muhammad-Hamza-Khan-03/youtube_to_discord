from sqlmodel import Session, create_engine, select
from main import ChannelAudit
from settings import settings

engine = create_engine(f"sqlite:///{settings.db_path}")

with Session(engine) as session:
    audits = session.exec(select(ChannelAudit)).all()
    for audit in audits:
        audit.is_circuit_broken = False
        audit.consecutive_errors = 0
        session.add(audit)
    session.commit()
    print(f"Reset {len(audits)} channels.")