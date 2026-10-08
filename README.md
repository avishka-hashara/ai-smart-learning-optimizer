# Smart Learning & Performance Optimizer

Students check in once a week with the hours they studied and how confident they feel in each subject. The app then runs four stages:

1. **Predict.** An MLP regressor predicts the exam mark for each subject.
2. **Assess risk.** A Mamdani fuzzy system compares confidence with the predicted mark and flags overconfidence or anxiety.
3. **Optimise.** A genetic algorithm reallocates the weekly hours toward the highest-priority subjects.
4. **Schedule.** A rule-based scheduler turns the hours into a weekly timetable, putting subjects with closer exams first.

Every check-in and its plan are saved, so students can look back at previous weeks and see how their predicted marks changed.

## Run it

From the project root, with the virtual environment active:

```bash
pip install -r requirements.txt
python src/prediction/prepare_data.py     # only if data/cleaned_features.csv is missing
python src/prediction/train_mlp.py        # only if models/predictor_v1.joblib is missing
uvicorn app.api:app --reload
```

Open http://localhost:8000 and create an account. The database (`smartlearn.db`, SQLite) is created and migrated on first start.

**No account?** The sign-in page has a "Try the demo" link (or go to `/#demo`). It opens the original persona dashboard, where nothing is saved. The Streamlit version is still available: `streamlit run app/main.py`.

## Configuration

Set these as environment variables (see `.env.example`):

| Variable | Default | Notes |
|---|---|---|
| `SECRET_KEY` | random per start | **Required in production.** Signs the session tokens. |
| `DATABASE_URL` | `sqlite:///./smartlearn.db` | Any SQLAlchemy URL, e.g. Postgres. |
| `COOKIE_SECURE` | `0` | Set to `1` behind HTTPS. |
| `SESSION_DAYS` | `7` | How long a sign-in lasts. |
| `AUTO_MIGRATE` | `1` | Runs `alembic upgrade head` at startup. |

## Data model

```
users ─┬─< subjects            name, difficulty (1–3), exam date, colour; archived, not deleted, once used
       └─< checkins            one per user per week (week_start = Monday)
              ├─< checkin_entries   hours + confidence per subject
              └── plans             full pipeline output (JSON) + model version
```

Submitting a check-in for a week that already has one replaces it. When a subject that appears in past check-ins is removed, it is archived so its history stays intact. Adding it again under the same name restores it.

After changing `app/models.py`, generate and apply a migration:

```bash
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

## API

Signed-in requests use the httpOnly `session` cookie that sign-in sets. Non-browser clients can send `Authorization: Bearer <token>` instead (the token is returned by register and login).

| Method | Path | |
|---|---|---|
| POST | `/api/auth/register` | `{email, password, name}` → creates the account and signs in |
| POST | `/api/auth/login` | `{email, password}` |
| POST | `/api/auth/logout` | |
| GET / DELETE | `/api/me` | Profile / permanently delete the account and all its data |
| GET | `/api/me/meta` | Input ranges, subject limit, current week |
| GET / POST | `/api/subjects` | List (`?include_archived=true`) / add |
| PATCH / DELETE | `/api/subjects/{id}` | Edit / remove (archived if it has history) |
| POST | `/api/checkins` | `{week_start?, entries: [{subject_id, hours, confidence}]}` → runs the pipeline, saves the plan |
| GET | `/api/checkins` | History summaries, newest week first |
| GET / DELETE | `/api/checkins/{id}` | Full check-in with its plan / delete |
| GET | `/api/meta`, POST `/api/run` | Public demo (personas), nothing saved |

Interactive docs are served at http://localhost:8000/docs.

## Tests

```bash
python -m pytest tests          # API: auth, isolation between users, check-ins, migrations
python -m tests.test_fuzzy      # stage tests (run as scripts)
python -m tests.test_optimizer
python smoke_test.py
```
