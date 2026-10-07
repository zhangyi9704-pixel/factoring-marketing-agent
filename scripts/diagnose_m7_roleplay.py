"""Independent model-only diagnosis; never executes or resumes a production graph."""
import argparse,json,logging,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);args=p.parse_args()
    from scripts.run_m7_validation import configure
    settings=configure(args.run);logging.disable(logging.CRITICAL)
    from app.agents.roleplay.agent import RoleplayAgent
    from app.engine.fact_cards import unlock_fact_cards
    from app.engine.leak_guard import check_reply,fingerprints
    from app.agents.roleplay.customer_reply import make_customer_reply_node
    from app.services.llm_service import LLMService
    full=RoleplayAgent()._snapshot("__m7_"+args.run+"_full").values
    state=dict(case=full["case"],turn=0,conversation=[m for m in full["conversation"] if m["turn"]==0],sales_input="我们需要资金周转支持，请先核实交易与回款情况。",released_info=[],unlocked_ids=[],dim_asked_count={},pending_directives=[],sensitive_fired=[])
    state.update(unlock_fact_cards(state))
    # External payload is explicitly constructed from public data only.
    # Complete case remains local solely for gate classification, never in the request.
    from app.schemas.case import CaseState,to_public_case,assert_no_internal_leak
    public=to_public_case(CaseState.model_validate(full["case"])).model_dump()
    assert_no_internal_leak(public)
    prompt=json.dumps(dict(public_case=public,sales_input=state["sales_input"]),ensure_ascii=False)
    observations=[]
    llm=LLMService(settings)
    for attempt in range(2):
        raw=llm.chat_json(prompt,model_role="chat",system="模拟客户。仅依据public_case作答，未知回答需核实。不得新增金额、编号或主体。仅输出JSON，键为reply、revealed（空数组）、mood。")
        reply=str(raw.get("reply", ""))
        reason=check_reply(state,reply).reasons
        observations.append(dict(reply_chars=len(reply),reasons=reason,expected_keys_present=all(k in raw for k in ("reply","revealed","mood"))))
        if not reason:break
    output=dict(run=args.run,observations=observations,public_payload_only=True,no_graph_or_database_write=True)
    path=ROOT/"docs/validation"/f"m7-roleplay-diagnosis-{args.run}.json"
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(output,ensure_ascii=False))

if __name__=="__main__":
    try:main()
    except Exception as exc:print("[FAIL] "+type(exc).__name__+"; details withheld");sys.exit(1)
