"""user profile fields

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-16 22:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('university', sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column('programme', sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column('year_of_study', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('password_changed_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('password_changed_at')
        batch_op.drop_column('year_of_study')
        batch_op.drop_column('programme')
        batch_op.drop_column('university')
