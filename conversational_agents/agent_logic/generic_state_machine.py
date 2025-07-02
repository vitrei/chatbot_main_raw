from transitions import Machine
import json
import os
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod

class GuardRailRule(ABC):
    """Abstract base class for guard rail rules"""
    
    @abstractmethod
    def check(self, current_state: str, dest_state: str, turn_counter: int, context: Dict[str, Any]) -> bool:
        """Check if transition is allowed"""
        pass
    
    @abstractmethod
    def get_reason(self) -> str:
        """Get reason for blocking transition"""
        pass

class SequenceRule(GuardRailRule):
    """Enforce mandatory sequence progression"""
    
    def __init__(self, sequence: List[str], allow_skip: bool = False, allow_backward: bool = False):
        self.sequence = sequence
        self.allow_skip = allow_skip
        self.allow_backward = allow_backward
    
    def check(self, current_state: str, dest_state: str, turn_counter: int, context: Dict[str, Any]) -> bool:
        try:
            current_idx = self.sequence.index(current_state)
            dest_idx = self.sequence.index(dest_state)
            
            # Backward movement check
            if dest_idx < current_idx and not self.allow_backward:
                return False
            
            # Skip check
            if dest_idx > current_idx + 1 and not self.allow_skip:
                return False
                
            return True
        except ValueError:
            # States not in sequence - allow transition
            return True
    
    def get_reason(self) -> str:
        return f"Sequence rule violation: must follow {' → '.join(self.sequence)}"

class TurnLimitRule(GuardRailRule):
    """Enforce turn-based limits with absolute closure at max turns"""
    
    def __init__(self, min_turns: Optional[int] = None, max_turns: Optional[int] = None, 
                 closure_states: Optional[List[str]] = None, force_closure: bool = False):
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.closure_states = closure_states or []
        self.force_closure = force_closure
    
    def check(self, current_state: str, dest_state: str, turn_counter: int, context: Dict[str, Any]) -> bool:
        print(f"  🔒 TurnLimitRule check: turn {turn_counter}, {current_state} → {dest_state}")
        
        # Early closure prevention
        if self.min_turns and turn_counter < self.min_turns and dest_state in self.closure_states:
            print(f"    ❌ Too early for closure (turn {turn_counter} < {self.min_turns})")
            return False
        
        # ABSOLUTE CLOSURE ENFORCEMENT AT TURN 12+
        if turn_counter >= 12:
            if dest_state in self.closure_states:
                print(f"    ✅ Closure allowed at turn {turn_counter}")
                return True
            else:
                print(f"    ❌ ABSOLUTE BLOCK: Turn {turn_counter} >= 12, only closure allowed")
                return False
        
        # Normal force closure logic (turn 11)
        if self.max_turns and turn_counter >= self.max_turns and self.force_closure:
            if dest_state not in self.closure_states:
                print(f"    ❌ Force closure enabled: only closure states allowed at turn {turn_counter}")
                return False
        
        print(f"    ✅ Turn limit check passed")
        return True
    
    def get_reason(self) -> str:
        return f"Turn limit rule: min {self.min_turns}, max {self.max_turns} turns"

class ProgressionRule(GuardRailRule):
    """Force progression every N turns"""
    
    def __init__(self, sequence: List[str], turns_per_state: int = 2):
        self.sequence = sequence
        self.turns_per_state = turns_per_state
    
    def check(self, current_state: str, dest_state: str, turn_counter: int, context: Dict[str, Any]) -> bool:
        # This rule doesn't block - it's used for forcing transitions
        return True
    
    def should_force_progression(self, current_state: str, turn_counter: int) -> Optional[str]:
        """Check if progression should be forced"""
        try:
            current_idx = self.sequence.index(current_state)
            expected_progression_turn = (current_idx + 1) * self.turns_per_state
            
            if turn_counter >= expected_progression_turn and current_idx < len(self.sequence) - 1:
                return self.sequence[current_idx + 1]
        except ValueError:
            pass
        
        return None
    
    def get_reason(self) -> str:
        return f"Progression rule: advance every {self.turns_per_state} turns"

class GoldenPathRule(GuardRailRule):
    """Enforce golden path with controlled derailing"""
    
    def __init__(self, golden_path: List[str], derailing_states: List[str], 
                 max_derailing_time: int = 3, force_return: bool = True):
        self.golden_path = golden_path
        self.derailing_states = derailing_states
        self.max_derailing_time = max_derailing_time
        self.force_return = force_return
        self.derailing_start_turn = None
    
    def check(self, current_state: str, dest_state: str, turn_counter: int, context: Dict[str, Any]) -> bool:
        print(f"  🎨 GoldenPathRule check: turn {turn_counter}, {current_state} → {dest_state}")
        
        # ALWAYS allow transitions to closure (override golden path rules near end)
        if dest_state == 'onboarding_closure' and turn_counter >= 10:
            print(f"    ✅ Closure override: allowing transition to closure at turn {turn_counter}")
            return True
        
        # Track derailing start
        if current_state in self.golden_path and dest_state in self.derailing_states:
            context['derailing_start_turn'] = turn_counter
            print(f"    🔄 Starting derailing: {current_state} → {dest_state}")
            return True
        
        # Force return to golden path after max derailing time
        if current_state in self.derailing_states and self.force_return:
            derailing_start = context.get('derailing_start_turn', turn_counter)
            derailing_duration = turn_counter - derailing_start
            
            if derailing_duration >= self.max_derailing_time:
                # Only allow transitions back to golden path or closure
                if dest_state not in self.golden_path and dest_state != 'onboarding_closure':
                    print(f"    ❌ Max derailing time exceeded: {derailing_duration} >= {self.max_derailing_time}")
                    return False
        
        print(f"    ✅ Golden path check passed")
        return True
    
    def get_reason(self) -> str:
        return f"Golden path rule: max {self.max_derailing_time} turns derailing allowed"

class AbsoluteClosureRule(GuardRailRule):
    """Absolute rule: ALWAYS allow transitions to closure after turn 10"""
    
    def __init__(self, closure_state: str = 'onboarding_closure', min_turn_for_closure: int = 10):
        self.closure_state = closure_state
        self.min_turn_for_closure = min_turn_for_closure
    
    def check(self, current_state: str, dest_state: str, turn_counter: int, context: Dict[str, Any]) -> bool:
        # This rule NEVER blocks transitions to closure after min turn
        if dest_state == self.closure_state and turn_counter >= self.min_turn_for_closure:
            print(f"  🆘 AbsoluteClosureRule: ALWAYS allowing closure at turn {turn_counter}")
            return True
        
        # This rule doesn't care about other transitions
        return True
    
    def get_reason(self) -> str:
        return f"Absolute closure rule: closure always allowed after turn {self.min_turn_for_closure}"

class GenericStateMachine:
    """Generic, configurable state machine for conversations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.guard_rails: List[GuardRailRule] = []
        self.turn_counter = 0
        
        # Extract basic configuration
        self.states = config.get('states', [])
        self.initial_state = config.get('initial_state', 'init')
        self.transitions_config = config.get('transitions', [])
        self.stages = config.get('stages', {})
        self.current_stage = config.get('current_stage', 'default')
        
        # Setup guard rails from configuration
        self._setup_guard_rails()
        
        # Create the machine
        self.machine = Machine(
            model=self,
            states=self.states,
            initial=self.initial_state,
            transitions=self.transitions_config
        )
        
        print(f"🎰 GENERIC STATE MACHINE INITIALIZED: {self.initial_state}")
    
    def _setup_guard_rails(self):
        """Setup guard rails from configuration"""
        stage_config = self.stages.get(self.current_stage, {})
        
        # Sequence rules
        if 'mandatory_sequence' in stage_config:
            sequence = stage_config['mandatory_sequence']
            guard_config = stage_config.get('guard_rails', {})
            
            self.guard_rails.append(SequenceRule(
                sequence=sequence,
                allow_skip=guard_config.get('allow_sequence_skip', False),
                allow_backward=guard_config.get('allow_backward', False)
            ))
        
        # Turn limit rules
        if 'min_turns' in stage_config or 'max_turns' in stage_config:
            self.guard_rails.append(TurnLimitRule(
                min_turns=stage_config.get('min_turns'),
                max_turns=stage_config.get('max_turns'),
                closure_states=stage_config.get('closure_states', ['onboarding_closure']),
                force_closure=stage_config.get('guard_rails', {}).get('force_closure_after_max_turns', False)
            ))
        
        # Progression rules
        if 'mandatory_sequence' in stage_config:
            sequence = stage_config['mandatory_sequence']
            turns_per_state = stage_config.get('progression_rules', {}).get('turns_per_state', 2)
            
            self.progression_rule = ProgressionRule(sequence, turns_per_state)
        
        # Absolute closure rule (HIGHEST PRIORITY - add first)
        closure_states = stage_config.get('closure_states', ['onboarding_closure'])
        if closure_states:
            self.guard_rails.append(AbsoluteClosureRule(
                closure_state=closure_states[0],
                min_turn_for_closure=10
            ))
        
        # Golden path rules
        if 'golden_path' in stage_config and 'derailing_states' in stage_config:
            golden_path = stage_config['golden_path']
            derailing_states = stage_config['derailing_states']
            max_derailing_time = stage_config.get('guard_rails', {}).get('max_derailing_time', 3)
            force_return = stage_config.get('guard_rails', {}).get('force_golden_path_return', True)
            
            self.guard_rails.append(GoldenPathRule(
                golden_path=golden_path,
                derailing_states=derailing_states,
                max_derailing_time=max_derailing_time,
                force_return=force_return
            ))
    
    def get_current_state(self) -> str:
        """Get current state"""
        return self.state
    
    def is_transition_allowed(self, dest_state: str, turn_counter: int, 
                            context: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
        """Check if transition is allowed by guard rails"""
        context = context or {}
        
        print(f"🔍 GUARD RAIL CHECK: {self.state} → {dest_state} (turn {turn_counter})")
        
        for i, rule in enumerate(self.guard_rails):
            rule_name = type(rule).__name__
            allowed = rule.check(self.state, dest_state, turn_counter, context)
            
            print(f"  Rule {i+1} ({rule_name}): {'✅ PASS' if allowed else '❌ BLOCK'}")
            
            if not allowed:
                reason = rule.get_reason()
                print(f"    ❌ BLOCKED BY: {reason}")
                return False, reason
        
        print(f"  ✅ ALL GUARD RAILS PASSED")
        return True, "Transition allowed"
    
    def get_available_transitions(self, turn_counter: int = 0, 
                                context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get available transitions filtered by guard rails"""
        available_triggers = []
        
        for transition in self.transitions_config:
            source = transition['source']
            if source == '*' or source == self.state or (isinstance(source, list) and self.state in source):
                dest_state = transition['dest']
                
                # Check guard rails
                allowed, reason = self.is_transition_allowed(dest_state, turn_counter, context)
                
                available_triggers.append({
                    'trigger': transition['trigger'],
                    'source': source,
                    'dest': dest_state,
                    'description': self.get_trigger_description(transition['trigger']),
                    'allowed': allowed,
                    'block_reason': reason if not allowed else None
                })
        
        return available_triggers
    
    def get_trigger_description(self, trigger: str) -> str:
        """Get description for trigger from config or default"""
        descriptions = self.config.get('trigger_descriptions', {})
        return descriptions.get(trigger, f"Trigger: {trigger}")
    
    def check_forced_transitions(self, turn_counter: int) -> Optional[str]:
        """Check if any transitions should be forced"""
        if hasattr(self, 'progression_rule'):
            return self.progression_rule.should_force_progression(self.state, turn_counter)
        return None
    
    def execute_transition(self, trigger_name: str, reason: str = "No reason provided", turn_counter: int = 0) -> bool:
        """Execute a state transition"""
        try:
            print(f"🚀 EXECUTING TRANSITION: {trigger_name} from {self.state}")
            
            # Get current available transitions with detailed info
            available_transitions = self.get_available_transitions(turn_counter)
            
            print(f"📋 Available transitions for {self.state}:")
            for t in available_transitions:
                status = '✅' if t['allowed'] else '❌'
                block_info = f" ({t['block_reason']})" if t['block_reason'] else ""
                print(f"  {status} {t['trigger']} → {t['dest']}{block_info}")
            
            # Check if trigger is valid and allowed
            matching_transition = next((t for t in available_transitions if t['trigger'] == trigger_name), None)
            
            if not matching_transition:
                print(f"❌ TRANSITION NOT FOUND: {trigger_name} not in available transitions")
                return False
            
            if not matching_transition['allowed']:
                print(f"❌ TRANSITION BLOCKED: {trigger_name} - {matching_transition['block_reason']}")
                return False
            
            old_state = self.state
            
            # Execute the trigger
            print(f"⚙️ Calling trigger method: {trigger_name}")
            trigger_method = getattr(self, trigger_name)
            trigger_method()
            
            print(f"✅ STATE MACHINE TRANSITION: {old_state} --{trigger_name}--> {self.state}")
            print(f"📝 TRANSITION REASON: {reason}")
            return True
            
        except AttributeError as e:
            print(f"❌ TRIGGER METHOD NOT FOUND: {trigger_name} - {str(e)}")
            return False
        except Exception as e:
            print(f"❌ STATE TRANSITION FAILED: {trigger_name} - {str(e)}")
            return False
    
    def get_state_context_for_decision_agent(self, turn_counter: int = 0) -> Dict[str, Any]:
        """Provide state context for decision agent"""
        available_transitions = self.get_available_transitions(turn_counter)
        
        # Filter only allowed transitions
        allowed_transitions = [t for t in available_transitions if t['allowed']]
        
        return {
            'current_state': self.state,
            'current_stage': self.current_stage,
            'available_transitions': allowed_transitions,
            'blocked_transitions': [t for t in available_transitions if not t['allowed']],
            'state_description': f"Current state: {self.state} (Stage: {self.current_stage})",
            'turn_counter': turn_counter,
            'forced_transition': self.check_forced_transitions(turn_counter)
        }

def load_state_machine_config(config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load state machine configuration from JSON file"""
    try:
        if not config_path:
            # Default path resolution
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
            config_path = os.path.join(project_root, 'prompts', 'state_machine.json')
            
            # Fallback path
            if not os.path.exists(config_path):
                config_path = "/home/merlotllm/Documents/project_scratch/chatbot_main_raw/prompts/state_machine.json"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            print(f"✅ Generic state machine config loaded from {config_path}")
            return config
            
    except Exception as e:
        print(f"❌ Error loading state machine config: {e}")
        return None

def create_generic_state_machine(config: Optional[Dict[str, Any]] = None) -> Optional[GenericStateMachine]:
    """Create a generic state machine from configuration"""
    try:
        if not config:
            config = load_state_machine_config()
        
        if not config:
            print("❌ No state machine config available")
            return None
        
        return GenericStateMachine(config)
        
    except Exception as e:
        print(f"❌ Error creating generic state machine: {e}")
        return None