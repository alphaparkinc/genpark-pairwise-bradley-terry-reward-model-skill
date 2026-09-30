import sys
import json
from client import BradleyTerryRewardModel

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-pairwise-bradley-terry-reward-model-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "evaluate_bradley_terry",
                        "description": "Calculates choice probability and loss for chosen vs rejected rewards",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "r_chosen": {"type": "number"},
                                "r_rejected": {"type": "number"}
                            },
                            "required": ["r_chosen", "r_rejected"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        
        if name == "evaluate_bradley_terry":
            rc = args.get("r_chosen", 0.0)
            rr = args.get("r_rejected", 0.0)
            res = {
                "prob": BradleyTerryRewardModel.predict_preference_prob(rc, rr),
                "loss": BradleyTerryRewardModel.compute_loss(rc, rr),
                "gradient_step": BradleyTerryRewardModel.gradient_step(rc, rr)
            }
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
            
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def run():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    run()
