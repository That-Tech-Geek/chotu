from economic_engine.agent_runtime.envelope import PolicyEnvelope
from economic_engine.agent_runtime.executor import AutonomousRuntime
from economic_engine.agent_runtime.idempotency import ActionExecutor
from economic_engine.agent_runtime.kill_switch import KillSwitch
from economic_engine.agent_runtime.state_machine import (
    NegotiationState,
    NegotiationStateMachine,
)
from economic_engine.state.canonical import (
    Deal,
    Merchant,
    Negotiation,
    NegotiationContext,
    Product,
    Supplier,
)


def ctx():
    return NegotiationContext(
        merchant=Merchant(id="m", name="m"),
        supplier=Supplier(id="s", merchant_id="m", name="s"),
        product=Product(id="p", merchant_id="m", sku="p", base_purchase_cost=100),
        negotiation=Negotiation(
            id="n", merchant_id="m", supplier_id="s", product_id="p", quantity=10,
        ),
    )


def test_envelope_blocks_price():
    env = PolicyEnvelope(max_unit_price=10.0, min_unit_price=1.0,
                         max_total_spend=100.0)
    d = Deal(price=20.0, quantity=1)
    assert any("max" in v for v in env.check_deal(d))


def test_kill_switch_state_hash():
    k = KillSwitch(max_price=100)
    assert k.state_hash(ctx()) == k.state_hash(ctx())


def test_state_machine_illegal_transition():
    sm = NegotiationStateMachine()
    try:
        sm.transition(NegotiationState.EVALUATING)
        assert False, "should raise"
    except ValueError:
        pass
    assert sm.state == NegotiationState.INIT


def test_idempotency_deduplicates():
    ex = ActionExecutor()
    d = Deal(price=1.0)
    r1 = ex.execute("n1", "a1", 0, d)
    r2 = ex.execute("n1", "a1", 0, d)
    assert r1["executed"] is True
    assert r2["duplicate"] is True


def test_autonomous_runtime_blocks_violation():
    env = PolicyEnvelope(max_unit_price=5.0, min_unit_price=0.0,
                         max_total_spend=10.0)
    kill = KillSwitch(max_price=5.0)
    rt = AutonomousRuntime(env, kill)
    d = Deal(price=10.0)
    out = rt.handle("n", "a", 0, d, ctx(), shadow=False)
    assert out["action"] == "BLOCKED"


def test_shadow_execution_recorded_not_sent():
    env = PolicyEnvelope(max_unit_price=1000.0, min_unit_price=0.0,
                         max_total_spend=1e6)
    kill = KillSwitch(max_price=1000.0)
    rt = AutonomousRuntime(env, kill)
    d = Deal(price=10.0)
    out = rt.handle("n", "a", 0, d, ctx(), shadow=True)
    assert out["action"] == "SHADOW"
    assert out["executed"] is False
