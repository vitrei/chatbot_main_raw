import json
import re
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages.ai import AIMessageChunk

from data_models.data_models import AgentState, NextActionDecision, NextActionDecisionType
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from large_language_models.llm_factory import llm_factory
from conversational_agents.agent_logic.state_machine_wrapper import create_state_machine_from_prompts, load_state_machine_config
from conversational_agents.agent_logic.general_logic.decision_rule_engine import DecisionRuleEngine
from conversational_agents.agent_logic.general_logic.decision_agent_helpers import DecisionAgentHelpers

from prompts.prompt_loader import prompt_loader

class LLMDecisionAgent(BaseDecisionAgent):

    def __init__(self):
        self.rule_engine = None
        super().__init__()
        
        decision_agent_prompt = """
Du bist Decision Agent. SCHNELL antworten!

=== KONTEXT ===
STATE: {current_state} | STAGE: {current_stage} | TURN: {turn_counter} ({stage_turn_info})
USER: {last_user_message}
PROFIL: {user_profile}

=== ZIELE ===
Stage: {stage_context_short}
State: {state_purpose_short}

=== TRANSITIONS ===
{available_transitions}

=== REGELN ===
{transition_logic}

=== ENTSCHEIDUNGSFRAMEWORK ===
1. User-Absicht verstehen:
   - "erzähl mir" = Vertiefung (meist kein Transition)
   - Themenwechsel = Transition wählen
   - Verwirrung = repair/comfort

2. Pädagogisches Ziel:
   - Vertiefung vs. Progression
   - User Engagement beachten

WICHTIG: Antworte NUR mit diesem JSON-Format:
{{
    "trigger": "exact_trigger_name",
    "reason": "Detaillierte Begründung basierend auf User-Absicht",
    "user_intent": "Was der User wirklich möchte",
    "pedagogical_goal": "Warum diese Transition das Lernziel unterstützt"
}}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Du bist ein intelligenter Decision Agent für Fake-News-Aufklärungsgespräche."),
            ("human", decision_agent_prompt),
        ])

        # Use agent-specific model from config
        from config import config
        decision_model = config.get("llm", "decision_agent_model", fallback=config.get("llm", "model_name"))
        llm = llm_factory.get_llm(decision_model)
        self.chain = prompt | llm 

    def add_state_machine_to_agent_state(self, agent_state):
        """Add state machine to existing AgentState"""
        
        if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
            state_machine = create_state_machine_from_prompts(agent_state.prompts)
            agent_state.state_machine = state_machine
            # print(f"State Machine added to AgentState: {state_machine.get_current_state() if state_machine else 'Failed'}")
        
        return agent_state
    
    def check_guard_rail_enforcement(self, agent_state: AgentState, available_transitions):
        """Check if guard rails require forced transitions using rule engine"""
        try:
            if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
                print("❌ No state machine available for guard rail check")
                return None
            
            # Initialize rule engine if not already done
            if self.rule_engine is None:
                config = load_state_machine_config()
                if config:
                    self.rule_engine = DecisionRuleEngine(config)
                    print(f"Rule engine initialized with {len(self.rule_engine.rules)} rules")
                else:
                    print("❌ Could not load config for rule engine")
                    return None
            
            # Get context for rule evaluation
            context = self.rule_engine.get_context_from_agent_state(agent_state, available_transitions)
            
            if 'error' in context:
                print(f"❌ Context error: {context['error']}")
                return None
            
            # Evaluate rules
            decision = self.rule_engine.evaluate_decision(context)
            
            if decision['forced']:
                print(f"RULE ENGINE DECISION: {decision['trigger']} (reason: {decision['reason']})")
                return decision['trigger']
            else:
                print(f"✅ No forced transitions needed - LLM can decide")
                return None
            
        except Exception as e:
            print(f"❌ Error in guard rail enforcement: {e}")
            return None
    

    def next_action(self, agent_state: AgentState):
        total_start = time.time()
        print(f"Turn: {agent_state.conversation_turn_counter}, User: {agent_state.user_id}")
        
        # Ensure state machine is initialized
        start_time = time.time()
        agent_state = self.add_state_machine_to_agent_state(agent_state)
        # print(f"State machine init: {time.time() - start_time:.3f}s")
        
        # Load comprehensive AgentState information
        start_time = time.time()
        agent_context = DecisionAgentHelpers.load_agent_state_context(agent_state)
        current_state = agent_context['current_state']
        # print(f"Agent context loading: {time.time() - start_time:.3f}s")
        
        # Get available transitions for current state
        start_time = time.time()
        available_transitions = DecisionAgentHelpers.get_current_state_transitions(agent_state)
        transition_logic = DecisionAgentHelpers.get_transition_decision_logic(agent_state, current_state)
        # print(f"Transitions & logic: {time.time() - start_time:.3f}s")
        
        # Check for guard rail enforcement
        start_time = time.time()
        forced_transition = self.check_guard_rail_enforcement(agent_state, available_transitions)
        # print(f"Guard rail check: {time.time() - start_time:.3f}s")
        if forced_transition:
            print(f"⚡ GUARD RAIL ENFORCEMENT: Forcing transition {forced_transition}")
            if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                success = agent_state.state_machine.execute_transition(forced_transition, "Guard rail enforcement", agent_state.conversation_turn_counter)
                if success:
                    current_state = agent_state.state_machine.get_current_state()
                    print(f"✅ FORCED TRANSITION EXECUTED: {forced_transition}")
                    # CRITICAL FIX: Reload transitions after forced transition
                    available_transitions = DecisionAgentHelpers.get_current_state_transitions(agent_state)
                    transition_logic = DecisionAgentHelpers.get_transition_decision_logic(agent_state, current_state)
                    print(f"🔄 RELOADED TRANSITIONS FOR NEW STATE: {current_state}")
        
        # Prepare FAST LLM prompt data - compact versions for speed
        start_time = time.time()
        current_stage = agent_state.state_machine.current_stage if hasattr(agent_state, 'state_machine') and agent_state.state_machine else 'unknown'
        stage_turn_info = DecisionAgentHelpers.get_stage_turn_info(agent_state)
        stage_context_short = DecisionAgentHelpers.get_stage_context_short(agent_state)
        state_purpose_short = DecisionAgentHelpers.get_state_purpose_short(agent_state, current_state)
        
        prompt_data = {
            "current_state": current_state,
            "current_stage": current_stage,
            "stage_turn_info": stage_turn_info,
            "last_user_message": agent_context['last_user_message'],
            "user_profile": agent_context['user_profile'],
            "turn_counter": agent_context['conversation_turn_counter'],
            "stage_context_short": stage_context_short,
            "state_purpose_short": state_purpose_short,
            "available_transitions": DecisionAgentHelpers.format_transitions_for_prompt(available_transitions),
            "transition_logic": transition_logic
        }
        # print(f"Prompt data prep: {time.time() - start_time:.3f}s")
        
        # LLM call to decide on transition - FAST MODE
        print("=== LLM CALL ===")
        print(f"{current_state}")
        
        try:
            start_time = time.time()

            response = self.chain.invoke(prompt_data)
            
            llm_time = time.time() - start_time
            
            # FAST JSON parsing - just get trigger
            parse_start = time.time()
            llm_decision = DecisionAgentHelpers.extract_trigger_fast(response.content)
            parse_time = time.time() - parse_start
            # print(f"JSON parsing: {parse_time:.3f}s")
            print(f"⚡ DECISION AGENT LLM: {llm_time:.2f}s")
            
            # Execute state transition (LLM MUST always choose one)
            transition_start = time.time()
            trigger = llm_decision.get('trigger')
            
            if not trigger:
                # Fallback: choose first allowed transition
                allowed_transitions_fallback = [t for t in available_transitions if t.get('allowed', True)]
                if allowed_transitions_fallback:
                    trigger = allowed_transitions_fallback[0]['trigger']
                    print(f"🔄 FALLBACK → {trigger}")
            
            if trigger:
                print(f"🔄 LLM → {trigger}")
                
                if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                    # Quick validation
                    allowed_transitions = [t for t in available_transitions if t.get('allowed', True)]
                    chosen_transition = next((t for t in allowed_transitions if t['trigger'] == trigger), None)
                    
                    if chosen_transition:
                        # Execute allowed transition
                        success = agent_state.state_machine.execute_transition(trigger, "LLM decision", agent_state.conversation_turn_counter)
                        if success:
                            current_state = agent_state.state_machine.get_current_state()
                    else:
                        # Handle blocked transitions
                        current_stage = agent_state.state_machine.current_stage
                        
                        # Special handling for stage_selection
                        if current_stage == 'stage_selection':
                            return NextActionDecision(
                                type=NextActionDecisionType.GENERATE_ANSWER, 
                                action="explain_unavailable_stage",
                                payload={"blocked_stage": trigger}
                            )
                        
                        # Fallback for other stages
                        if allowed_transitions:
                            fallback_trigger = allowed_transitions[0]['trigger']
                            agent_state.state_machine.execute_transition(fallback_trigger, "Fallback", agent_state.conversation_turn_counter)
                            current_state = agent_state.state_machine.get_current_state()
            else:
                print(f"🚫 NO TRANSITION POSSIBLE for {current_state}")
            
        except Exception as e:
            print(f"❌ LLM Decision failed: {e}")
            # Exception fallback: use first allowed transition
            allowed_transitions_exception = [t for t in available_transitions if t.get('allowed', True)]
            if allowed_transitions_exception:
                trigger = allowed_transitions_exception[0]['trigger']
                reason = "Exception fallback - LLM failed"
                print(f"🔄 EXCEPTION FALLBACK TRANSITION: {trigger}")
                if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                    success = agent_state.state_machine.execute_transition(trigger, reason, agent_state.conversation_turn_counter)
                    if success:
                        current_state = agent_state.state_machine.get_current_state()
                        print(f"✅ EXCEPTION FALLBACK EXECUTED: {trigger}")
                    else:
                        print(f"❌ EXCEPTION FALLBACK FAILED: {trigger}")
            else:
                print(f"❌ NO ALLOWED TRANSITIONS FOR EXCEPTION FALLBACK")
                print(f"    Available transitions: {[t['trigger'] for t in available_transitions]}")
                print(f"    All blocked by guard rails")
            llm_decision = {'guiding_instruction': 'general_guidance'}
            trigger = None  # Ensure trigger is set for safety
        
        # Load state-specific content from state_machine.json
        config_start = time.time()
        state_machine_config = load_state_machine_config()
        current_state_prompts = []
        current_state_examples = []
        
        if state_machine_config:
            state_prompts = state_machine_config.get('state_system_prompts', {})
            state_examples = state_machine_config.get('state_examples', {})
            current_state_prompts = state_prompts.get(current_state, [])
            current_state_examples = state_examples.get(current_state, [])
        
        config_time = time.time() - config_start
        transition_time = time.time() - transition_start if 'transition_start' in locals() else 0
        total_time = time.time() - total_start
        
        # print(f"Transition execution: {transition_time:.3f}s")
        # print(f"Config loading: {config_time:.3f}s") 
        # print(f"DECISION AGENT TOTAL: {total_time:.2f}s")
            
            # print(f"LOADED STATE PROMPTS: {len(current_state_prompts)} for {current_state}")
            # print(f"LOADED STATE EXAMPLES: {len(current_state_examples)} for {current_state}")
        
        # Return PROMPT_ADAPTION with all context
        payload = {
            # State Machine Context
            'current_state': current_state,
            'state_system_prompts': current_state_prompts,
            'state_examples': current_state_examples,
            
            # AgentState Context
            'user_profile': agent_context['user_profile'],
            'conversation_turn_counter': agent_context['conversation_turn_counter'],
            'user_id': agent_context['user_id'],
            'last_user_message': agent_context['last_user_message'],
            
            # Additional Context
            'available_transitions': available_transitions,
            'stage_info': agent_context['stage_info'],
            'fake_news_available': agent_context.get('fake_news_available', False),
            'fake_news_url': agent_context.get('fake_news_stimulus_url'),
            
            # Guiding instruction from LLM decision
            'guiding_instruction': llm_decision.get('guiding_instruction', 'general_guidance')
        }
        
        # Debug: Show complete payload being returned
        print(f"📦 PROMPT_ADAPTION PAYLOAD:")
        print(f"  🎯 current_state: {payload['current_state']}")
        print(f"  📝 state_system_prompts: {len(payload['state_system_prompts'])} items")
        for i, prompt in enumerate(payload['state_system_prompts'][:3], 1):
            print(f"    {i}. {prompt[:80]}...")
        print(f"  📚 state_examples: {len(payload['state_examples'])} items")
        print(f"  👤 user_profile: {payload['user_profile'][:80]}...")
        print(f"  🎭 guiding_instruction: {payload['guiding_instruction']}")
        print(f"  🔄 available_transitions: {len(payload['available_transitions'])} items")
        
        return NextActionDecision(
            type=NextActionDecisionType.PROMPT_ADAPTION,
            action="inject_complete_context",
            payload=payload
        )

    

