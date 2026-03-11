create table if not exists job_runs (
  run_id        text primary key,
  job_type      text not null,
  source        text not null,
  status        text not null,
  started_at    timestamptz not null,
  ended_at      timestamptz,
  duration_ms   bigint,
  progress_pct  int,
  message       text,
  records_count bigint default 0,
  pages_count   bigint default 0,
  errors_count  bigint default 0,
  records_per_sec numeric(10, 3),
  params_json   jsonb,
  result_json   jsonb,
  error_text    text,
  updated_at    timestamptz not null
);

create index if not exists idx_job_runs_started_at on job_runs(started_at desc);
create index if not exists idx_job_runs_status on job_runs(status);
create index if not exists idx_job_runs_source on job_runs(source);
create index if not exists idx_job_runs_records_per_sec on job_runs(records_per_sec);
