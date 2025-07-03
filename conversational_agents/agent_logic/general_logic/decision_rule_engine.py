"""
Decision Rule Engine for Generic State Machine
Handles configurable transition decision logic
"""

from typing import Dict, List, Any, Optional, Callable
import json


class DecisionRule:
    """Base class for decision rules"""
    
    def __init__(self, name: str, priority: int = 100):
        self.name = name
        self.priority = priority
    
    def applies(self, context: Dict[str, Any]) -> bool:
        """Check if this rule applies to the current context"""
        return True
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        """Evaluate rule and return trigger name if applicable"""
        return None
    
    def get_reason(self, context: Dict[str, Any]) -> str:
        """Get human-readable reason for the decision"""
        return f"Rule {self.name} applied"


class ClosureRule(DecisionRule):
    """Force closure after certain conditions"""
    
    def __init__(self, closure_trigger: str, conditions: Dict[str, Any]):
        super().__init__("closure_rule", priority=10)  # High priority
        self.closure_trigger = closure_trigger
        self.conditions = conditions
    
    def applies(self, context: Dict[str, Any]) -> bool:
        stage = context.get('current_stage', '')
        state = context.get('current_state', '')
        
        # Check if stage matches
        target_stages = self.conditions.get('stages', [])
        if target_stages and stage not in target_stages:
            return False
        
        # Check if state matches  
        target_states = self.conditions.get('states', [])
        if target_states and state not in target_states:
            return False
            
        return True
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        turn_counter = context.get('turn_counter', 0)
        stage_relative_turns = context.get('stage_relative_turns', 0)
        available_transitions = context.get('available_transitions', [])
        max_turns_in_stage = context.get('max_turns_in_stage', 15)
        
        # Check various closure conditions
        if self.conditions.get('absolute_turn_limit') and turn_counter >= self.conditions['absolute_turn_limit']:
            closure_transitions = [t for t in available_transitions if t['trigger'] == self.closure_trigger]
            if closure_transitions:
                return self.closure_trigger
        
        if self.conditions.get('stage_turn_limit') and stage_relative_turns >= self.conditions['stage_turn_limit']:
            closure_transitions = [t for t in available_transitions if t['trigger'] == self.closure_trigger]
            if closure_transitions:
                return self.closure_trigger
        
        if self.conditions.get('max_turns_ratio') and stage_relative_turns >= max_turns_in_stage * self.conditions['max_turns_ratio']:
            closure_transitions = [t for t in available_transitions if t['trigger'] == self.closure_trigger]
            if closure_transitions:
                return self.closure_trigger
        
        return None
    
    def get_reason(self, context: Dict[str, Any]) -> str:
        return f"Closure rule triggered: {self.closure_trigger}"


class ProgressionRule(DecisionRule):
    """Force progression between specific states"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("progression_rule", priority=20)
        self.config = config
    
    def applies(self, context: Dict[str, Any]) -> bool:
        current_state = context.get('current_state', '')
        source_states = self.config.get('source_states', [])
        return current_state in source_states
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        current_state = context.get('current_state', '')
        stage_relative_turns = context.get('stage_relative_turns', 0)
        available_transitions = context.get('available_transitions', [])
        
        # Check if we should force progression
        min_turns = self.config.get('min_turns_in_state', 1)
        max_turns = self.config.get('max_turns_in_state', 3)
        
        if stage_relative_turns >= min_turns:
            # Look for target transitions
            target_patterns = self.config.get('target_patterns', [])
            for pattern in target_patterns:
                matching_transitions = [t for t in available_transitions if pattern in t['dest']]
                if matching_transitions:
                    return matching_transitions[0]['trigger']
        
        return None
    
    def get_reason(self, context: Dict[str, Any]) -> str:
        return f"Progression rule: moving from {context.get('current_state', 'unknown')} after {context.get('stage_relative_turns', 0)} turns"


class CyclingPreventionRule(DecisionRule):
    """Prevent cycling between states"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("cycling_prevention", priority=15)
        self.config = config
    
    def applies(self, context: Dict[str, Any]) -> bool:
        current_state = context.get('current_state', '')
        cycling_states = self.config.get('cycling_states', [])
        return current_state in cycling_states
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        stage_relative_turns = context.get('stage_relative_turns', 0)
        available_transitions = context.get('available_transitions', [])
        
        # Force progression if cycling detected
        max_cycling_turns = self.config.get('max_cycling_turns', 6)
        if stage_relative_turns >= max_cycling_turns:
            escape_patterns = self.config.get('escape_patterns', [])
            for pattern in escape_patterns:
                matching_transitions = [t for t in available_transitions if pattern in t['dest']]
                if matching_transitions:
                    return matching_transitions[0]['trigger']
        
        return None
    
    def get_reason(self, context: Dict[str, Any]) -> str:
        return f"Cycling prevention: escaping after {context.get('stage_relative_turns', 0)} turns"


class StageSelectionRule(DecisionRule):
    """Handle stage selection logic with dynamic stage availability"""
    
    def __init__(self):
        super().__init__("stage_selection_rule", priority=25)
    
    def applies(self, context: Dict[str, Any]) -> bool:
        return context.get('current_stage') == 'stage_selection'
    
    def _detect_unavailable_stage_request(self, context: Dict[str, Any]) -> Optional[str]:
        """Detect if user is asking for an unavailable stage"""
        # Get user's last message (this would need to be passed in context)
        last_message = context.get('last_user_message', '').lower()
        completed_stages = context.get('completed_stages', [])
        
        # Check if user is asking for psychology/society and it's completed
        if any(word in last_message for word in ['psychologie', 'psychology', 'gesellschaft', 'society', 'social media']) and 'content_psychology_society' in completed_stages:
            return 'content_psychology_society'
        
        # Check if user is asking for politics/tech and it's completed  
        if any(word in last_message for word in ['politik', 'politics', 'technologie', 'technology', 'deepfake', 'ki']) and 'content_politics_tech' in completed_stages:
            return 'content_politics_tech'
        
        return None
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        available_content_stages = context.get('available_content_stages', [])
        all_stages_completed = context.get('all_stages_completed', False)
        available_transitions = context.get('available_transitions', [])
        completed_stages = context.get('completed_stages', [])
        
        # Check if user is requesting an unavailable stage
        unavailable_stage_requested = self._detect_unavailable_stage_request(context)
        if unavailable_stage_requested:
            print(f"🚫 User requesting unavailable stage: {unavailable_stage_requested}")
            # Set a special context flag for the LLM to handle this
            context['unavailable_stage_requested'] = unavailable_stage_requested
            # Let LLM handle with special prompt
            return None
        
        # Filter available transitions to only show those that are allowed
        allowed_transitions = [t for t in available_transitions if t.get('allowed', True)]
        
        # If all content stages are completed, force offboarding
        if all_stages_completed:
            offboarding_transitions = [t for t in allowed_transitions if t['trigger'] == 'finish_all_content']
            if offboarding_transitions:
                print(f"🎯 All content stages completed - forcing offboarding")
                return 'finish_all_content'
        
        # Check if there are no allowed content stage transitions left
        content_transitions = [t for t in allowed_transitions if t['trigger'].startswith('choose_')]
        if len(content_transitions) == 0 and len(completed_stages) > 0:
            # Only offboarding is available
            offboarding_transitions = [t for t in allowed_transitions if t['trigger'] == 'finish_all_content']
            if offboarding_transitions:
                print(f"🎯 No more content stages available - suggesting offboarding")
                return 'finish_all_content'
        
        # Otherwise, let LLM choose from available content stages
        print(f"🎯 Stage selection: {len(available_content_stages)} stages available, {len(content_transitions)} transitions allowed")
        return None
    
    def get_reason(self, context: Dict[str, Any]) -> str:
        return "Stage selection rule: guiding stage choice based on completion status"


class DecisionRuleEngine:
    """Engine that evaluates decision rules in priority order"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules: List[DecisionRule] = []
        self._setup_rules_from_config()
    
    def _setup_rules_from_config(self):
        """Setup rules from configuration"""
        decision_rules_config = self.config.get('decision_rules', {})
        
        # Add closure rules
        for rule_config in decision_rules_config.get('closure_rules', []):
            self.rules.append(ClosureRule(
                closure_trigger=rule_config['trigger'],
                conditions=rule_config['conditions']
            ))
        
        # Add progression rules
        for rule_config in decision_rules_config.get('progression_rules', []):
            self.rules.append(ProgressionRule(rule_config))
        
        # Add cycling prevention rules
        for rule_config in decision_rules_config.get('cycling_prevention_rules', []):
            self.rules.append(CyclingPreventionRule(rule_config))
        
        # Add stage selection rule
        self.rules.append(StageSelectionRule())
        
        # Sort by priority (lower number = higher priority)
        self.rules.sort(key=lambda r: r.priority)
    
    def evaluate_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all applicable rules and return decision"""
        # print(f"🎯 RULE ENGINE: Evaluating {len(self.rules)} rules")
        
        for rule in self.rules:
            if rule.applies(context):
                # print(f"  📋 Rule {rule.name} (priority {rule.priority}) applies")
                trigger = rule.evaluate(context)
                if trigger:
                    reason = rule.get_reason(context)
                    print(f"  ✅ Rule {rule.name} decided: {trigger}")
                    return {
                        'trigger': trigger,
                        'reason': reason,
                        'rule_name': rule.name,
                        'forced': True
                    }
                else:
                    pass
                    # print(f"  ⏭️ Rule {rule.name} applied but no trigger returned")
        
        print(f"  🤷 No rules triggered - LLM decision needed")
        return {
            'trigger': None,
            'reason': 'No forced rules triggered',
            'rule_name': None,
            'forced': False
        }
    
    def get_context_from_agent_state(self, agent_state, available_transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract context from agent state for rule evaluation"""
        if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
            return {'error': 'No state machine available'}
        
        sm = agent_state.state_machine
        turn_counter = agent_state.conversation_turn_counter
        stage_start_turn = getattr(sm, 'stage_start_turn', 0)
        stage_relative_turns = turn_counter - stage_start_turn
        
        # Get stage configuration
        stage_config = sm.stages.get(sm.current_stage, {})
        
        context = {
            'current_state': sm.get_current_state(),
            'current_stage': sm.current_stage,
            'turn_counter': turn_counter,
            'stage_relative_turns': stage_relative_turns,
            'stage_start_turn': stage_start_turn,
            'available_transitions': available_transitions,
            'max_turns_in_stage': stage_config.get('max_turns_in_stage', 15),
            'target_turns': stage_config.get('target_turns', 12),
            'closure_states': stage_config.get('closure_states', []),
            'golden_path': stage_config.get('golden_path', []),
            'derailing_states': stage_config.get('derailing_states', []),
            'last_user_message': getattr(agent_state, 'instruction', ''),
            'available_content_stages': getattr(sm, 'completed_stages', []),
            'completed_stages': getattr(sm, 'completed_stages', [])
        }
        
        return context