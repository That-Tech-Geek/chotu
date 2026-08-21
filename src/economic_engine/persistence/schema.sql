-- Supabase / Postgres schema (RLS-enabled). Tenant = merchant_id.
create table if not exists merchants (id text primary key, name text);
create table if not exists suppliers (id text primary key, merchant_id text, name text, reliability_history numeric default 0.5);
create table if not exists products (id text primary key, merchant_id text, sku text, base_purchase_cost numeric);
create table if not exists transactions (id text primary key, merchant_id text, product_id text, quantity numeric, unit_price numeric);
create table if not exists inventory (product_id text primary key, merchant_id text, on_hand numeric, reserved numeric);

create table if not exists negotiations (id text primary key, merchant_id text, supplier_id text, product_id text, quantity numeric, status text, agreed_price numeric, reservation_price numeric, created_at timestamptz default now());
create table if not exists negotiation_rounds (negotiation_id text, round_index int, price numeric, quantity numeric, actor text, response text, created_at timestamptz default now());
create table if not exists negotiation_messages (negotiation_id text, actor text, content text, created_at timestamptz default now());
create table if not exists offers (id text primary key, negotiation_id text, price numeric, quantity numeric, payment_terms_days int, delivery_days int, actor text);
create table if not exists decisions (id text primary key, negotiation_id text, action text, price numeric, confidence numeric, cvar_95 numeric, expected_profit numeric, strategy text, created_at timestamptz default now());
create table if not exists outcomes (decision_id text primary key, actual_profit numeric, actual_price numeric, responded text, created_at timestamptz default now());

create table if not exists supplier_profiles (supplier_id text, posterior jsonb, updated_at timestamptz default now());
create table if not exists relationship_profiles (supplier_id text, merchant_id text, lifetime_value numeric, interaction_count int, reputation numeric, primary key (supplier_id, merchant_id));

create table if not exists strategy_population (generation int, strategy text, weight numeric, primary key (generation, strategy));
create table if not exists strategy_fitness (generation int, strategy text, fitness numeric, primary key (generation, strategy));

create table if not exists feature_snapshots (id text primary key, negotiation_id text, features jsonb, created_at timestamptz default now());
create table if not exists model_predictions (id text primary key, negotiation_id text, model_version text, prediction float);
create table if not exists model_outcomes (prediction_id text primary key, actual float, created_at timestamptz default now());
create table if not exists model_versions (version text primary key, promoted_at timestamptz, metrics jsonb);

create table if not exists simulation_runs (id text primary key, negotiation_id text, mode text, samples int, summary jsonb, created_at timestamptz default now());
create table if not exists learning_events (id text primary key, negotiation_id text, predicted numeric, actual numeric, error numeric, created_at timestamptz default now());
create table if not exists audit_events (id text primary key, event_type text, payload jsonb, created_at timestamptz default now());
create table if not exists index_versions (version int, created_at timestamptz default now(), description text);
create table if not exists negotiation_embeddings (id text primary key, negotiation_id text, embedding vector(128), created_at timestamptz default now());

-- Durable idempotency: one row per executed action key. Unique constraint is
-- the cross-process guarantee against duplicate external side effects.
create table if not exists action_records (
    idempotency_key text primary key,
    payload jsonb,
    created_at timestamptz default now()
);

alter table action_records enable row level security;

-- fingerprint column distinguishes claim owners for the upsert-then-verify pattern
alter table action_records add column if not exists fingerprint text;

-- RLS policies: tenant-scoped by merchant_id.
alter table merchants enable row level security;
alter table suppliers enable row level security;
alter table products enable row level security;
alter table transactions enable row level security;
alter table inventory enable row level security;
alter table negotiations enable row level security;
alter table negotiation_rounds enable row level security;
alter table negotiation_messages enable row level security;
alter table offers enable row level security;
alter table decisions enable row level security;
alter table outcomes enable row level security;
alter table supplier_profiles enable row level security;
alter table relationship_profiles enable row level security;
alter table strategy_population enable row level security;
alter table strategy_fitness enable row level security;
alter table feature_snapshots enable row level security;
alter table model_predictions enable row level security;
alter table model_outcomes enable row level security;
alter table model_versions enable row level security;
alter table simulation_runs enable row level security;
alter table learning_events enable row level security;
alter table audit_events enable row level security;
alter table index_versions enable row level security;
alter table negotiation_embeddings enable row level security;
