#!/usr/bin/env python3
"""
Quick test script for the new rule engine
"""

import sys
import os
sys.path.append('/home/merlotllm/Documents/project_scratch/chatbot_main_raw')

from conversational_agents.agent_logic.general_logic.decision_rule_engine import DecisionRuleEngine
from conversational_agents.agent_logic.state_machine_wrapper import load_state_machine_config

def test_rule_engine():
    print("🧪 TESTING RULE ENGINE")
    
    # Load config
    config = load_state_machine_config()
    if not config:
        print("❌ Failed to load config")
        return
    
    # Create rule engine
    rule_engine = DecisionRuleEngine(config)
    print(f"✅ Rule engine created with {len(rule_engine.rules)} rules")
    
    # Test case 1: Content synthesis state should trigger closure
    test_context_1 = {
        'current_state': 'content_synthesis_pt',
        'current_stage': 'content_politics_tech',
        'turn_counter': 15,
        'stage_relative_turns': 7,
        'stage_start_turn': 8,
        'available_transitions': [
            {'trigger': 'finish_content', 'dest': 'content_closure_pt'},
            {'trigger': 'other_trigger', 'dest': 'somewhere_else'}
        ],
        'max_turns_in_stage': 10,
        'target_turns': 8,
        'closure_states': ['content_closure_pt']
    }
    
    print("\n🧪 TEST 1: Content synthesis should trigger closure")
    decision_1 = rule_engine.evaluate_decision(test_context_1)
    print(f"Result: {decision_1}")
    
    # Test case 2: Deep dive state should trigger synthesis
    test_context_2 = {
        'current_state': 'psychology_deep_dive',
        'current_stage': 'content_psychology_society',
        'turn_counter': 20,
        'stage_relative_turns': 7,
        'stage_start_turn': 13,
        'available_transitions': [
            {'trigger': 'synthesize_content', 'dest': 'content_synthesis_ps'},
            {'trigger': 'explore_society', 'dest': 'society_deep_dive'}
        ],
        'max_turns_in_stage': 10,
        'target_turns': 8,
        'closure_states': ['content_closure_ps']
    }
    
    print("\n🧪 TEST 2: Deep dive cycling should trigger synthesis")
    decision_2 = rule_engine.evaluate_decision(test_context_2)
    print(f"Result: {decision_2}")
    
    # Test case 3: Normal state - no forced rules
    test_context_3 = {
        'current_state': 'content_intro_pt',
        'current_stage': 'content_politics_tech',
        'turn_counter': 10,
        'stage_relative_turns': 2,
        'stage_start_turn': 8,
        'available_transitions': [
            {'trigger': 'start_politics', 'dest': 'politics_deep_dive'},
            {'trigger': 'start_technology', 'dest': 'technology_deep_dive'}
        ],
        'max_turns_in_stage': 10,
        'target_turns': 8,
        'closure_states': ['content_closure_pt']
    }
    
    print("\n🧪 TEST 3: Normal state - should allow LLM decision")
    decision_3 = rule_engine.evaluate_decision(test_context_3)
    print(f"Result: {decision_3}")

if __name__ == "__main__":
    test_rule_engine()