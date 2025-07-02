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
    """Enforce turn-based limits with optional stage-relative counting"""
    
    def __init__(self, min_turns: Optional[int] = None, max_turns: Optional[int] = None, 
                 closure_states: Optional[List[str]] = None, force_closure: bool = False,
                 stage_relative: bool = False, max_turns_in_stage: Optional[int] = None):
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.closure_states = closure_states or []
        self.force_closure = force_closure
        self.stage_relative = stage_relative
        self.max_turns_in_stage = max_turns_in_stage
    
    def check(self, current_state: str, dest_state: str, turn_counter: int, context: Dict[str, Any]) -> bool:
        # Get stage context for relative counting
        stage_start_turn = context.get('stage_start_turn', 0)
        turns_in_stage = turn_counter - stage_start_turn
        current_stage = context.get('current_stage', 'onboarding')
        
        if self.stage_relative and self.max_turns_in_stage:
            print(f"  🔒 TurnLimitRule (stage-relative): turn {turn_counter} (stage: {turns_in_stage}/{self.max_turns_in_stage}), {current_state} → {dest_state}")
            
            # Stage-relative logic: only check turns within current stage
            if turns_in_stage > self.max_turns_in_stage and dest_state in self.closure_states:
                print(f"    ❌ Stage turn limit exceeded: {turns_in_stage} > {self.max_turns_in_stage}")
                return False
            
            print(f"    ✅ Stage-relative turn limit check passed")
            return True
        else:
            # Original absolute turn logic (only for onboarding stage)
            print(f"  🔒 TurnLimitRule (absolute): turn {turn_counter}, {current_state} → {dest_state}")
            
            # Early closure prevention
            if self.min_turns and turn_counter < self.min_turns and dest_state in self.closure_states:
                print(f"    ❌ Too early for closure (turn {turn_counter} < {self.min_turns})")
                return False
            
            # ABSOLUTE CLOSURE ENFORCEMENT AT TURN 12+ (ONLY for onboarding stage)
            if turn_counter >= 12 and self.closure_states and current_stage == 'onboarding':
                if dest_state in self.closure_states:
                    print(f"    ✅ Closure allowed at turn {turn_counter}")
                    return True
                # EXCEPTION: Allow inter-stage transitions even at turn 12+
                elif dest_state in ['stage_selection', 'content_intro_pt', 'content_intro_ps']:
                    print(f"    ✅ Inter-stage transition allowed at turn {turn_counter}: {dest_state}")
                    return True
                else:
                    print(f"    ❌ ABSOLUTE BLOCK: Turn {turn_counter} >= 12, only closure/inter-stage allowed (onboarding stage)")
                    return False
            
            # Normal force closure logic
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
        self.stage_start_turn = 0  # Track when current stage started
        
        # Extract basic configuration
        self.stages = config.get('stages', {})
        self.current_stage = config.get('current_stage', 'default')
        self.initial_state = config.get('initial_state', 'init')
        self.transitions_config = config.get('transitions', [])
        
        # Start with current stage states only
        current_stage_config = self.stages.get(self.current_stage, {})
        self.states = current_stage_config.get('states', [])
        
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
        print(f"🎭 CURRENT STAGE: {self.current_stage}")
        print(f"📋 STAGE STATES: {self.states}")
    
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
        if 'min_turns' in stage_config or 'max_turns' in stage_config or stage_config.get('relative_turns'):
            stage_relative = stage_config.get('relative_turns', False)
            max_turns_in_stage = stage_config.get('max_turns_in_stage')
            
            self.guard_rails.append(TurnLimitRule(
                min_turns=stage_config.get('min_turns'),
                max_turns=stage_config.get('max_turns'),
                closure_states=stage_config.get('closure_states', ['onboarding_closure']),
                force_closure=stage_config.get('guard_rails', {}).get('force_closure_after_max_turns', False),
                stage_relative=stage_relative,
                max_turns_in_stage=max_turns_in_stage
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
        
        # Add stage context for guard rails
        context['stage_start_turn'] = getattr(self, 'stage_start_turn', 0)
        context['current_stage'] = self.current_stage
        
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
            
            # SPECIAL HANDLING for inter-stage transitions
            if trigger_name == "proceed_to_stage_selection":
                return self._execute_inter_stage_transition("stage_selection", turn_counter, reason)
            elif trigger_name == "choose_politics_tech":
                return self._execute_inter_stage_transition("content_politics_tech", turn_counter, reason, "content_intro_pt")
            elif trigger_name == "choose_psychology_society":
                return self._execute_inter_stage_transition("content_psychology_society", turn_counter, reason, "content_intro_ps")
            
            # Normal intra-stage transitions
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
            
            # Check if method exists and is callable
            if not hasattr(self, trigger_name):
                print(f"❌ Trigger method {trigger_name} not found on object")
                print(f"   Available methods: {[m for m in dir(self) if not m.startswith('_')]}")
                return False
            
            trigger_method = getattr(self, trigger_name)
            if not callable(trigger_method):
                print(f"❌ {trigger_name} is not callable")
                return False
                
            print(f"✅ Trigger method {trigger_name} found and callable")
            
            # Try to execute the trigger
            try:
                trigger_method()
            except Exception as trigger_exception:
                print(f"❌ Trigger execution failed: {trigger_exception}")
                # Let's try to manually update the state as a fallback
                dest_state = matching_transition['dest']
                print(f"🔄 FALLBACK: Manually updating state from {self.state} to {dest_state}")
                self.state = dest_state
                print(f"✅ FALLBACK STATE UPDATE: {old_state} → {self.state}")
                return True
            
            print(f"✅ STATE MACHINE TRANSITION: {old_state} --{trigger_name}--> {self.state}")
            print(f"📝 TRANSITION REASON: {reason}")
            
            return True
            
        except AttributeError as e:
            print(f"❌ TRIGGER METHOD NOT FOUND: {trigger_name} - {str(e)}")
            return False
        except Exception as e:
            print(f"❌ STATE TRANSITION FAILED: {trigger_name} - {str(e)}")
            return False
    
    def _execute_inter_stage_transition(self, target_stage: str, turn_counter: int, reason: str, target_state: str = None) -> bool:
        """Execute inter-stage transition manually"""
        try:
            old_state = self.state
            old_stage = self.current_stage
            
            # Determine target state
            if target_state is None:
                # For proceed_to_stage_selection, the target state is stage_selection itself
                target_state = "stage_selection"
            
            print(f"🎭 INTER-STAGE TRANSITION: {old_stage}:{old_state} → {target_stage}:{target_state}")
            
            # Switch stage and recreate state machine
            success = self.switch_stage(target_stage, turn_counter)
            if not success:
                return False
            
            # Verify target state exists in new stage
            if target_state not in self.states:
                print(f"❌ Target state '{target_state}' not in new stage states: {self.states}")
                target_state = self.states[0] if self.states else 'unknown'
                print(f"🔄 Using fallback state: {target_state}")
            
            # The switch_stage method sets the state to the initial state of the new stage
            # For inter-stage transitions, we often want a specific target state
            print(f"🎯 Current state after stage switch: {self.state}")
            print(f"🎯 Desired target state: {target_state}")
            
            if target_state != self.state:
                print(f"🎯 Need to update state from {self.state} to {target_state}")
                # Use the transitions library to properly set the state
                if hasattr(self.machine, 'set_state'):
                    self.machine.set_state(target_state)
                    print(f"✅ Set state using machine.set_state: {self.state}")
                else:
                    # Fallback to direct assignment
                    self.state = target_state
                    print(f"✅ Set state using direct assignment: {self.state}")
            else:
                print(f"✅ State already correct: {self.state}")
            
            print(f"✅ INTER-STAGE TRANSITION COMPLETED: {old_stage}:{old_state} → {target_stage}:{target_state}")
            print(f"📝 TRANSITION REASON: {reason}")
            print(f"🔍 VERIFICATION: Current state is now {self.get_current_state()}")
            print(f"🔍 VERIFICATION: Current stage is now {self.current_stage}")
            print(f"🔍 VERIFICATION: Available states: {self.states}")
            print(f"🔍 VERIFICATION: Stage start turn: {self.stage_start_turn}")
            
            return True
            
        except Exception as e:
            print(f"❌ INTER-STAGE TRANSITION FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_inter_stage_transition(self):
        """Check and execute inter-stage transitions"""
        try:
            inter_stage_config = self.config.get('inter_stage_transitions', {})
            current_state = self.state
            
            if current_state in inter_stage_config:
                stage_transition = inter_stage_config[current_state]
                
                if isinstance(stage_transition, str):
                    # Direct stage transition (e.g., "onboarding_closure": "stage_selection")
                    if current_state == "onboarding_closure":
                        print(f"🔄 INTER-STAGE: Onboarding completed → proceeding to stage_selection")
                        # Switch to stage_selection stage and execute transition
                        # Note: We need the current turn counter here, but it's not passed to this method
                        # This will be handled by the manual stage switch in the decision agent
                        self.proceed_to_stage_selection()
                        
                elif isinstance(stage_transition, dict):
                    # Choice-based transition (handled by decision agent)
                    print(f"🔀 INTER-STAGE CHOICE: {current_state} allows multiple stage transitions")
                    # This will be handled by the decision agent choosing the appropriate trigger
                    
        except Exception as e:
            print(f"❌ Error in inter-stage transition check: {e}")
    
    def switch_stage(self, new_stage: str, current_turn: int = 0):
        """Switch to a new stage and update state machine configuration"""
        try:
            if new_stage not in self.config.get('stages', {}):
                print(f"❌ Stage '{new_stage}' not found in configuration")
                return False
            
            old_stage = self.current_stage
            old_state = self.state
            
            print(f"🎭 STAGE SWITCH REQUEST: {old_stage}:{old_state} → {new_stage}")
            
            self.current_stage = new_stage
            
            # Track when this stage started for relative turn counting
            self.stage_start_turn = current_turn
            
            # Update states for new stage
            new_stage_config = self.stages.get(new_stage, {})
            self.states = new_stage_config.get('states', [])
            
            # Filter transitions for new stage
            new_stage_transitions = [t for t in self.transitions_config 
                                   if self._transition_belongs_to_stage(t, new_stage)]
            
            # Recreate the state machine with new stage configuration
            initial_state = self.states[0] if self.states else self.initial_state
            
            print(f"🔧 RECREATING MACHINE: initial_state={initial_state}, states={self.states}")
            print(f"🔧 TRANSITIONS FOR STAGE: {len(new_stage_transitions)} transitions")
            for t in new_stage_transitions:
                print(f"    {t['trigger']}: {t['source']} → {t['dest']}")
            
            self.machine = Machine(
                model=self,
                states=self.states,
                initial=initial_state,
                transitions=new_stage_transitions
            )
            
            # Force the state to be set correctly - the Machine should initialize to initial_state
            # but we need to ensure it's properly set
            self.state = initial_state
            print(f"🔧 MACHINE STATE AFTER CREATION: {getattr(self, 'state', 'NOT_SET')}")
            
            # Verify the machine knows about our current state
            if hasattr(self.machine, 'model') and hasattr(self.machine.model, 'state'):
                print(f"🔧 MACHINE MODEL STATE: {self.machine.model.state}")
            
            # Test if trigger methods are available
            for t in new_stage_transitions:
                trigger_name = t['trigger']
                if hasattr(self, trigger_name):
                    print(f"✅ Trigger method available: {trigger_name}")
                else:
                    print(f"❌ Trigger method missing: {trigger_name}")
            
            # Verify machine states
            print(f"🔧 MACHINE STATES: {getattr(self.machine, 'states', 'NOT_FOUND')}")
            if hasattr(self.machine, 'models'):
                models = self.machine.models
                if isinstance(models, dict):
                    print(f"🔧 MACHINE MODELS (dict): {list(models.keys())}")
                elif isinstance(models, list):
                    print(f"🔧 MACHINE MODELS (list): {len(models)} models")
                else:
                    print(f"🔧 MACHINE MODELS (other): {type(models)} - {models}")
            
            # Force refresh of dynamic methods if possible
            if hasattr(self.machine, '_add_model_to_state'):
                for state in self.states:
                    try:
                        self.machine._add_model_to_state(self, state)
                        print(f"✅ Added model to state: {state}")
                    except Exception as e:
                        print(f"❌ Failed to add model to state {state}: {e}")
            
            # Clear old guard rails and setup new ones
            self.guard_rails.clear()
            self._setup_guard_rails()
            
            print(f"🎭 STAGE SWITCH: {old_stage} → {new_stage}")
            print(f"📋 Stage started at turn: {self.stage_start_turn}")
            print(f"📋 New stage states: {self.states}")
            print(f"🔄 New stage transitions: {len(new_stage_transitions)}")
            print(f"🔒 New guard rails: {len(self.guard_rails)} rules loaded")
            
            return True
            
        except Exception as e:
            print(f"❌ Error switching stage: {e}")
            return False
    
    def _transition_belongs_to_stage(self, transition: Dict[str, Any], stage: str) -> bool:
        """Check if a transition belongs to a specific stage"""
        stage_config = self.stages.get(stage, {})
        stage_states = stage_config.get('states', [])
        
        source = transition.get('source')
        dest = transition.get('dest')
        trigger = transition.get('trigger')
        
        # Check if both source and dest are in stage states
        belongs = False
        if isinstance(source, str):
            belongs = source in stage_states and dest in stage_states
        elif isinstance(source, list):
            belongs = any(s in stage_states for s in source) and dest in stage_states
        
        # Minimal debug logging for performance
        if stage.startswith('content_') and not belongs:
            print(f"🔍 FILTER: {trigger} for {stage}: ❌")
        
        return belongs
    
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