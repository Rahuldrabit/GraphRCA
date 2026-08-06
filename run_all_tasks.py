import os
import json
import logging
from datetime import datetime
from aiopslab.orchestrator import Orchestrator
from aiopslab.service.factory import ServiceFactory
from swarm_agent_aiopslab import SwarmGraphRCAAgent
from GraphRCA_agent.agent_aiopslab import GraphRCAAgent

logger = logging.getLogger(__name__)

def run_all_tasks():
    # Setup global logging
    os.makedirs("results", exist_ok=True)
    report_file = f"results/full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Initialize AIOpsLab orchestrator
    orchestrator = Orchestrator()
    service = ServiceFactory.create_service() # Defaults to reading AIOpsLab config
    
    # Get all benchmarks/tasks
    # Assume we get tasks via AIOpsLab API
    tasks = service.get_all_tasks() if hasattr(service, "get_all_tasks") else []
    
    if not tasks:
        # Fallback to known test scenarios if API doesn't expose it simply
        tasks = [
            {"id": "detection_1", "type": "detection", "desc": "namespace: test-hotel-reservation"},
            {"id": "localization_1", "type": "localization", "desc": "namespace: astronomy-shop"},
            {"id": "analysis_1", "type": "analysis", "desc": "namespace: social-network"},
            {"id": "mitigation_1", "type": "mitigation", "desc": "namespace: test-hotel-reservation"}
        ]
        
    mode = os.getenv("GRAPHRCA_AGENT_MODE", "pipeline")
    AgentClass = SwarmGraphRCAAgent if mode == "scratchpad_swarm" else GraphRCAAgent
    
    results = []
    
    for task in tasks:
        task_id = task.get("id")
        task_type = task.get("type")
        desc = task.get("desc")
        
        logger.info(f"Starting task: {task_id} ({task_type})")
        
        output_dir = f"results/{task_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        agent = AgentClass(
            problem_desc=desc,
            task_type=task_type,
            output_dir=output_dir,
            verbose=True
        )
        
        # Start agent
        agent.run()
        
        # In a real environment we'd call orchestrator.evaluate(agent)
        # Assuming the orchestrator evaluates and handles generator communication
        try:
            # Fake/mock execution for demonstration, real execution requires
            # orchestrator integration loop
            agent.finalize()
            
            # Extract final metrics
            res = {
                "task_id": task_id,
                "task_type": task_type,
                "agent_mode": mode,
                "result": agent.result
            }
            results.append(res)
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Full report generated at {report_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_tasks()
