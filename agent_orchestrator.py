# =====================================================================
#  NOVAMART MULTI-AGENT CUSTOMER SUPPORT SYSTEM
# =====================================================================
#  Completed implementation for Tasks 2, 3, 4, 6 and deployment.
# =====================================================================

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from strands_agents import Agent, BedrockModel
from agent_observability import tool
import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base

dynamodb = boto3.resource("dynamodb", region_name=config.AWS_REGION)
orders_table = dynamodb.Table(config.ORDERS_TABLE)
customers_table = dynamodb.Table(config.CUSTOMERS_TABLE)
state_table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

def _get_worker_model(temperature: float = 0.1) -> BedrockModel:
    model = BedrockModel(model_id=config.WORKER_MODEL_ID, temperature=temperature)
    return _apply_guardrail(model)

def _get_orchestrator_model(temperature: float = 0.0) -> BedrockModel:
    model = BedrockModel(model_id=config.ORCHESTRATOR_MODEL_ID, temperature=temperature)
    return _apply_guardrail(model)

def build_inventory_agent() -> Agent:
    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """Look up order record from DynamoDB using composite key (customer_id, order_id)."""
        resp = orders_table.get_item(Key={"customer_id": customer_id, "order_id": order_id})
        return resp.get("Item", {"status": "NOT_FOUND"})

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """Retrieve customer membership tier and profile from DynamoDB."""
        resp = customers_table.get_item(Key={"customer_id": customer_id})
        return resp.get("Item", {"tier": "Standard"})

    @tool
    def list_customer_orders(customer_id: str) -> list:
        """List all historical orders placed by a specific customer."""
        from boto3.dynamodb.conditions import Key
        resp = orders_table.query(KeyConditionExpression=Key("customer_id").eq(customer_id))
        return resp.get("Items", [])

    return Agent(
        model=_get_worker_model(temperature=0.1),
        system_prompt="You are the Inventory Agent. Gather order and customer facts from DynamoDB accurately. Never make refund decisions.",
        tools=[check_order_status, get_customer_tier, list_customer_orders]
    )

def build_refund_agent() -> Agent:
    @tool
    def get_inventory_context(session_id: str) -> dict:
        """Read accumulated inventory facts from the shared DynamoDB WorkflowState."""
        state = _read_workflow_state(session_id)
        return state.get("inventory_findings", {})

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """Process a return by updating the order record in DynamoDB."""
        resp = orders_table.update_item(
            Key={"customer_id": customer_id, "order_id": order_id},
            UpdateExpression="SET #st = :r, refund_reason = :rs",
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={":r": "RETURN_INITIATED", ":rs": reason},
            ReturnValues="ALL_NEW"
        )
        return resp.get("Attributes", {"status": "RETURN_INITIATED"})

    return Agent(
        model=_get_worker_model(temperature=0.1),
        system_prompt="You are the Refund Agent. Evaluate refund eligibility (Standard = 30 days, Premium = 60 days) and initiate refunds.",
        tools=[get_inventory_context, initiate_refund]
    )

def build_policy_agent() -> Agent:
    @tool
    def retrieve_returns(query: str) -> str:
        """Retrieve return policy passages from Knowledge Base."""
        return retrieve_from_knowledge_base(query, config.RETURNS_KB_ID)

    @tool
    def retrieve_shipping(query: str) -> str:
        """Retrieve shipping policy passages from Knowledge Base."""
        return retrieve_from_knowledge_base(query, config.SHIPPING_KB_ID)

    @tool
    def retrieve_warranty(query: str) -> str:
        """Retrieve warranty policy passages from Knowledge Base."""
        return retrieve_from_knowledge_base(query, config.WARRANTY_KB_ID)

    returns_agent = Agent(model=_get_worker_model(temperature=0.0), system_prompt="Returns retriever.", tools=[retrieve_returns])
    shipping_agent = Agent(model=_get_worker_model(temperature=0.0), system_prompt="Shipping retriever.", tools=[retrieve_shipping])
    warranty_agent = Agent(model=_get_worker_model(temperature=0.0), system_prompt="Warranty retriever.", tools=[retrieve_warranty])

    @tool
    def search_all_policies(query: str) -> dict:
        """Search returns, shipping, and warranty policies concurrently using ThreadPoolExecutor."""
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_map = {
                executor.submit(returns_agent, query): "returns",
                executor.submit(shipping_agent, query): "shipping",
                executor.submit(warranty_agent, query): "warranty"
            }
            for future in as_completed(future_map):
                domain = future_map[future]
                try:
                    results[domain] = str(future.result())
                except Exception as e:
                    results[domain] = f"Error: {str(e)}"
        return results

    return Agent(
        model=_get_worker_model(temperature=0.2),
        system_prompt="You are the Policy Agent coordinator. Always call search_all_policies first, then synthesize passages.",
        tools=[search_all_policies]
    )

def build_communication_agent() -> Agent:
    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """Read the complete WorkflowState record containing all previous findings."""
        return _read_workflow_state(session_id)

    return Agent(
        model=_get_worker_model(temperature=0.3),
        system_prompt="You are the Customer Communication Specialist. Draft warm, comprehensive final customer responses.",
        tools=[get_full_workflow_context]
    )

def build_orchestrator_agent() -> Agent:
    inventory_agent = build_inventory_agent()
    policy_agent = build_policy_agent()
    refund_agent = build_refund_agent()
    communication_agent = build_communication_agent()

    @tool
    def initialize_session(session_id: str, customer_id: str) -> dict:
        """Initialize a new WorkflowState record in DynamoDB. Must be called first."""
        return _create_workflow_state(session_id, customer_id)

    @tool
    def route_to_inventory_agent(session_id: str, query: str) -> str:
        """Route customer inquiry to Inventory Agent."""
        state = _read_workflow_state(session_id)
        result = str(inventory_agent(query))
        _update_workflow_state(session_id, {"inventory_findings": result}, state.get("version", 0))
        return result

    @tool
    def route_to_policy_agent(session_id: str, query: str) -> str:
        """Route policy inquiries to Policy Agent."""
        state = _read_workflow_state(session_id)
        result = str(policy_agent(query))
        _update_workflow_state(session_id, {"policy_findings": result}, state.get("version", 0))
        return result

    @tool
    def route_to_refund_agent(session_id: str, query: str) -> str:
        """Route return/refund requests to Refund Agent."""
        state = _read_workflow_state(session_id)
        result = str(refund_agent(query))
        _update_workflow_state(session_id, {"refund_findings": result}, state.get("version", 0))
        return result

    @tool
    def route_to_communication_agent(session_id: str, query: str) -> str:
        """Always call this tool last to draft the final response."""
        return str(communication_agent(query))

    system_prompt = (
        "You are the NovaMart Support Orchestrator. Follow these rules strictly:\n"
        "1. Every request: ALWAYS call initialize_session first.\n"
        "2. Order status / return / refund: Route to Inventory first, then Refund.\n"
        "3. Policy questions: Route to Policy Agent.\n"
        "4. Account questions: Route to Inventory Agent ONLY.\n"
        "5. Math/calculation: Route directly to CommunicationAgent.\n"
        "6. Every request: ALWAYS finish by routing to route_to_communication_agent."
    )

    return Agent(
        model=_get_orchestrator_model(temperature=0.0),
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent
        ]
    )

def create_guardrail():
    bedrock = boto3.client("bedrock", region_name=config.AWS_REGION)
    response = bedrock.create_guardrail(
        name="novamart-enterprise-guardrail",
        description="Enterprise guardrail for NovaMart support",
        topicPolicyConfig={
            "topicsConfig": [
                {"name": "CompetitorProducts", "definition": "Mentions or endorsements of rival competitor platforms.", "type": "DENY"},
                {"name": "PricingNegotiation", "definition": "Haggling or requesting custom discounts on advertised pricing.", "type": "DENY"},
                {"name": "LegalThreats", "definition": "Threats of legal action, lawsuits, or attorneys.", "type": "DENY"}
            ],
            "tierConfig": {"tierName": "STANDARD"}
        },
        contentPolicyConfig={
            "filtersConfig": [
                {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
                {"type": "MISCONDUCT", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"}
            ]
        },
        wordPolicyConfig={"managedWordListsConfig": [{"type": "PROFANITY"}]},
        sensitiveInformationPolicyConfig={
            "piiEntitiesConfig": [
                {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
                {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"},
                {"type": "EMAIL", "action": "ANONYMIZE"},
                {"type": "PHONE", "action": "ANONYMIZE"}
            ]
        },
        blockedInputMessaging="Query violates NovaMart security and customer policy.",
        blockedOutputsMessaging="Cannot fulfill request under NovaMart guidelines.",
        crossRegionConfig={"guardrailProfileIdentifier": "us.guardrail.v1:0"}
    )
    guardrail_id = response["guardrailId"]
    ver_res = bedrock.create_guardrail_version(guardrailIdentifier=guardrail_id)
    return guardrail_id, ver_res["version"]

def deploy_to_agentcore_runtime(guardrail_id: str, guardrail_version: str):
    import agentcore_cli
    runtime_env = {
        "AWS_REGION": config.AWS_REGION,
        "PROJECT_NAME": config.PROJECT_NAME,
        "RETURNS_KB_ID": config.RETURNS_KB_ID,
        "SHIPPING_KB_ID": config.SHIPPING_KB_ID,
        "WARRANTY_KB_ID": config.WARRANTY_KB_ID,
        "AGENT_LOG_GROUP": config.AGENT_LOG_GROUP,
        "GUARDRAIL_ID": guardrail_id,
        "GUARDRAIL_VERSION": str(guardrail_version)
    }
    agentcore_cli.configure_runtime(network_mode="PUBLIC", protocol="HTTP", execution_role_arn=config.AGENTCORE_ROLE_ARN, env_vars=runtime_env)
    agentcore_cli.deploy()
    return agentcore_cli.deployed_runtime_arn()

def configure_memory(runtime_arn: str):
    agentcore_control = boto3.client("bedrock-agentcore-control", region_name=config.AWS_REGION)
    response = agentcore_control.create_memory(
        name="NovaMartSessionSummaryMemory",
        description="Session summary memory store with 7 day retention",
        memoryStrategy="summaryMemoryStrategy",
        eventExpiryDuration=7
    )
    memory_arn = response["memoryArn"]
    waiter = agentcore_control.get_waiter("memory_active")
    waiter.wait(memoryArn=memory_arn)
    return memory_arn

def configure_observability(runtime_arn: str):
    logging_config = {
        "cloudWatchConfig": {"logGroupName": config.AGENT_LOG_GROUP, "logLevel": "INFO", "enabled": True},
        "xRayConfig": {"enabled": True, "samplingRate": 1.0}
    }
    try:
        apply_observability_config(runtime_arn, logging_config)
    except Exception as e:
        print(f"[Observability Warning] {e}")

def deploy():
    print("[1/4] Creating Guardrail...")
    gid, gver = create_guardrail()
    print("[2/4] Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(gid, gver)
    print("[3/4] Configuring Memory...")
    mem_arn = configure_memory(runtime_arn)
    print("[4/4] Configuring Observability...")
    configure_observability(runtime_arn)
    print(f"Deployment complete!\nRuntime ARN: {runtime_arn}\nMemory ARN: {mem_arn}")
