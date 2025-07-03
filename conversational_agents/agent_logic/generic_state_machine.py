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
            # Stage-relative logic: only check turns within current stage
            if turns_in_stage > self.max_turns_in_stage and dest_state in self.closure_states:
                return False
            return True
        else:
            # Original absolute turn logic (only for onboarding stage)
            # Early closure prevention
            if self.min_turns and turn_counter < self.min_turns and dest_state in self.closure_states:
                return False
            
            # ABSOLUTE CLOSURE ENFORCEMENT AT TURN 12+ (ONLY for onboarding stage)
            if turn_counter >= 12 and self.closure_states and current_stage == 'onboarding':
                if dest_state in self.closure_states:
                    return True
                # EXCEPTION: Allow inter-stage transitions even at turn 12+
                elif dest_state in ['stage_selection', 'content_intro_pt', 'content_intro_ps']:
                    return True
                else:
                    return False
            
            # Normal force closure logic
            if self.max_turns and turn_counter >= self.max_turns and self.force_closure:
                if dest_state not in self.closure_states:
                    print(f"    ❌ Force closure enabled: only closure states allowed at turn {turn_counter}")
                    return False
            
            # print(f"    ✅ Turn limit check passed")
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
        # print(f"  🎨 GoldenPathRule check: turn {turn_counter}, {current_state} → {dest_state}")
        
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
        
        # print(f"    ✅ Golden path check passed")
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
        print("=== STATE MACHINE SERVICE ===")
        self.config = config
        self.guard_rails: List[GuardRailRule] = []
        self.turn_counter = 0
        self.stage_start_turn = 0  # Track when current stage started
        self.completed_stages: List[str] = []  # Track completed content stages
        
        # PERFORMANCE: Cache guard rail results
        self._transition_cache = {}
        self._last_cache_turn = -1
        
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
        
        # print(f"🎰 GENERIC STATE MACHINE INITIALIZED: {self.initial_state}")
        # print(f"🎭 CURRENT STAGE: {self.current_stage}")
        # print(f"📋 STAGE STATES: {self.states}")
    
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
        
        # Check guard rails (minimal logging)
        for rule in self.guard_rails:
            if not rule.check(self.state, dest_state, turn_counter, context):
                print(f"🚫 BLOCKED: {self.state} → {dest_state} ({rule.get_reason()})")
                return False, rule.get_reason()
        return True, "Transition allowed"
    
    def get_available_transitions(self, turn_counter: int = 0, 
                                context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get available transitions filtered by guard rails - WITH CACHING"""
        
        # PERFORMANCE: Use cache if same turn and state
        cache_key = f"{self.state}_{turn_counter}_{self.current_stage}"
        if cache_key in self._transition_cache and turn_counter == self._last_cache_turn:
            return self._transition_cache[cache_key]
        available_triggers = []
        
        for transition in self.transitions_config:
            source = transition['source']
            if source == '*' or source == self.state or (isinstance(source, list) and self.state in source):
                dest_state = transition['dest']
                trigger = transition['trigger']
                
                # Check if this transition leads to a completed stage (and block it)
                stage_blocked = False
                stage_block_reason = None
                
                if self.current_stage == 'stage_selection':
                    # Get target stage from inter_stage_transitions config
                    inter_stage_config = self.config.get('inter_stage_transitions', {})
                    stage_selection_config = inter_stage_config.get('stage_selection', {})
                    
                    target_stage = stage_selection_config.get(trigger)
                    if target_stage and target_stage.startswith('content_') and target_stage in self.completed_stages:
                        stage_blocked = True
                        stage_block_reason = f"Stage {target_stage} already completed in this session"
                        # Block completed stages silently
                
                # Check guard rails
                allowed, reason = self.is_transition_allowed(dest_state, turn_counter, context)
                
                # Combine stage blocking with guard rail blocking
                if stage_blocked:
                    allowed = False
                    reason = stage_block_reason
                
                available_triggers.append({
                    'trigger': trigger,
                    'source': source,
                    'dest': dest_state,
                    'description': self.get_trigger_description(trigger),
                    'allowed': allowed,
                    'block_reason': reason if not allowed else None
                })
        
        # PERFORMANCE: Cache results
        self._transition_cache[cache_key] = available_triggers
        self._last_cache_turn = turn_counter
        
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
            # print(f"🚀 EXECUTING TRANSITION: {trigger_name} from {self.state}")
            
            # SPECIAL HANDLING for inter-stage transitions
            inter_stage_mappings = {
                "proceed_to_stage_selection": ("stage_selection", "stage_selection"),
                "choose_politics_tech": ("content_politics_tech", "content_intro_pt"),
                "choose_psychology_society": ("content_psychology_society", "content_intro_ps"),
                "finish_all_content": ("offboarding", "stage_completion_review")
            }
            
            if trigger_name in inter_stage_mappings:
                target_stage, target_state = inter_stage_mappings[trigger_name]
                return self._execute_inter_stage_transition(target_stage, turn_counter, reason, target_state)
            
            # Normal intra-stage transitions
            available_transitions = self.get_available_transitions(turn_counter)
            print("=== AVAILABLE TRANSITIONS ===")
            print(f"{self.state}:")
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
            print("=== Calling trigger method:===")
            print(f"{trigger_name}")
            
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
                # print(f"🔄 FALLBACK: Manually updating state from {self.state} to {dest_state}")
                self.state = dest_state
                
                # PERFORMANCE: Clear cache after state change
                self._transition_cache.clear()
                
                # print(f"✅ FALLBACK STATE UPDATE: {old_state} → {self.state}")
                return True
            
            # print(f"✅ STATE MACHINE TRANSITION: {old_state} --{trigger_name}--> {self.state}")
            # print(f"📝 TRANSITION REASON: {reason}")
            
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
            
            # Mark current stage as completed if it's a content stage being left
            if old_stage.startswith('content_') and target_stage != old_stage:
                self.mark_stage_completed(old_stage)
            
            # Switch stage and recreate state machine
            success = self.switch_stage(target_stage, turn_counter)
            if not success:
                return False
            
            # Verify target state exists in new stage
            if target_state not in self.states:
                # print(f"❌ Target state '{target_state}' not in new stage states: {self.states}")
                target_state = self.states[0] if self.states else 'unknown'
                # print(f"🔄 Using fallback state: {target_state}")
            
            # The switch_stage method sets the state to the initial state of the new stage
            # For inter-stage transitions, we often want a specific target state
            # print(f"🎯 Current state after stage switch: {self.state}")
            # print(f"🎯 Desired target state: {target_state}")
            
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
            # print(f"🔍 VERIFICATION: Current state is now {self.get_current_state()}")
            # print(f"🔍 VERIFICATION: Current stage is now {self.current_stage}")
            # print(f"🔍 VERIFICATION: Available states: {self.states}")
            # print(f"🔍 VERIFICATION: Stage start turn: {self.stage_start_turn}")
            
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
            
            # Recreating state machine silently
            
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
            
            # Setup models silently
            if hasattr(self.machine, '_add_model_to_state'):
                for state in self.states:
                    try:
                        self.machine._add_model_to_state(self, state)
                    except:
                        pass
            
            # Clear old guard rails and setup new ones
            self.guard_rails.clear()
            self._setup_guard_rails()
            
            print(f"🎭 STAGE SWITCH: {old_stage} → {new_stage}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error switching stage: {e}")
            return False
    
    def mark_stage_completed(self, stage_name: str):
        """Mark a stage as completed"""
        if stage_name not in self.completed_stages and stage_name.startswith('content_'):
            self.completed_stages.append(stage_name)
            print(f"📋 STAGE COMPLETED: {stage_name} (Total completed: {len(self.completed_stages)})")
    
    def get_available_content_stages(self) -> List[str]:
        """Get list of content stages that haven't been completed yet"""
        all_content_stages = [name for name in self.stages.keys() if name.startswith('content_')]
        available_stages = [stage for stage in all_content_stages if stage not in self.completed_stages]
        print(f"📊 STAGE STATUS: Available={available_stages}, Completed={self.completed_stages}")
        return available_stages
    
    def _determine_other_content_stage(self, turn_counter: int) -> tuple[str, str]:
        """Determine which content stage to transition to based on current stage"""
        current_stage = self.current_stage
        
        if current_stage == "content_politics_tech":
            # User completed politics/tech, now show psychology/society
            return ("content_psychology_society", "content_intro_ps")
        elif current_stage == "content_psychology_society":
            # User completed psychology/society, now show politics/tech
            return ("content_politics_tech", "content_intro_pt")
        else:
            # Fallback - shouldn't happen but just in case
            print(f"⚠️ Unexpected stage for choose_other_content: {current_stage}")
            return ("content_politics_tech", "content_intro_pt")
    
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
            pass
            # Silent filtering
        
        return belongs
    
    def get_state_context_for_decision_agent(self, turn_counter: int = 0) -> Dict[str, Any]:
        """Provide state context for decision agent"""
        available_transitions = self.get_available_transitions(turn_counter)
        
        # Filter only allowed transitions
        allowed_transitions = [t for t in available_transitions if t['allowed']]
        
        context = {
            'current_state': self.state,
            'current_stage': self.current_stage,
            'available_transitions': allowed_transitions,
            'blocked_transitions': [t for t in available_transitions if not t['allowed']],
            'state_description': f"Current state: {self.state} (Stage: {self.current_stage})",
            'turn_counter': turn_counter,
            'forced_transition': self.check_forced_transitions(turn_counter)
        }
        
        # Add stage completion info if in stage_selection
        if self.current_stage == 'stage_selection':
            available_content_stages = self.get_available_content_stages()
            context['available_content_stages'] = available_content_stages
            context['completed_stages'] = self.completed_stages.copy()
            context['all_stages_completed'] = len(available_content_stages) == 0
            
            # Generate dynamic stage selection prompt based on available stages
            context['dynamic_stage_prompt'] = self._generate_stage_selection_prompt(allowed_transitions, context)
        
        return context
    
    def _generate_stage_selection_prompt(self, allowed_transitions: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """Generate dynamic prompt for stage_selection based on available transitions"""
        content_transitions = [t for t in allowed_transitions if t['trigger'].startswith('choose_')]
        offboarding_transitions = [t for t in allowed_transitions if t['trigger'] == 'finish_all_content']
        context = context or {}
        
        # Check if user requested an unavailable stage
        unavailable_stage = context.get('unavailable_stage_requested')
        if unavailable_stage:
            stage_name_map = {
                'content_psychology_society': 'Psychologie & Gesellschaft',
                'content_politics_tech': 'Politik & Technologie'
            }
            stage_display_name = stage_name_map.get(unavailable_stage, unavailable_stage)
            
            prompt_parts = [
                f"Ah, über {stage_display_name} haben wir bereits ausführlich gesprochen! 😊",
                f"Wir haben schon viel über die {stage_display_name.lower()} bei Fake News gelernt.",
                "Lass uns lieber zu einem neuen Bereich wechseln:"
            ]
            
            # Add remaining available options
            if any(t['trigger'] == 'choose_politics_tech' for t in content_transitions):
                prompt_parts.append("Option: 'Politik & Technologie - Wahlen, Deepfakes, KI-Tools'")
            
            if any(t['trigger'] == 'choose_psychology_society' for t in content_transitions):
                prompt_parts.append("Option: 'Psychologie & Gesellschaft - Manipulation, Social Media, Auswirkungen'")
            
            if offboarding_transitions:
                prompt_parts.append("Option: 'Gespräch beenden und alles zusammenfassen'")
            
            prompt_parts.append("Was wäre für dich interessant?")
            return " ".join(prompt_parts)
        
        if len(content_transitions) == 0:
            # No content stages available - only offboarding
            return (
                "Du hast alle verfügbaren Content-Bereiche abgeschlossen! "
                "Zeit für eine abschließende Reflexion über das Gelernte. "
                "Frage: 'Sollen wir das Gespräch zusammenfassen und beenden?'"
            )
        
        prompt_parts = ["Biete dem User folgende verfügbare Content-Bereiche an:"]
        
        # Add available content stages
        if any(t['trigger'] == 'choose_politics_tech' for t in content_transitions):
            prompt_parts.append("Option: 'Politik & Technologie - Wahlen, Deepfakes, KI-Tools'")
        
        if any(t['trigger'] == 'choose_psychology_society' for t in content_transitions):
            prompt_parts.append("Option: 'Psychologie & Gesellschaft - Manipulation, Social Media, Auswirkungen'")
        
        # Add offboarding option
        if offboarding_transitions:
            prompt_parts.append("Option: 'Gespräch beenden und Lernerfahrung reflektieren'")
        
        # Add completion info if some stages are already done
        if len(self.completed_stages) > 0:
            completed_names = []
            for stage in self.completed_stages:
                if stage == 'content_politics_tech':
                    completed_names.append('Politik & Technologie')
                elif stage == 'content_psychology_society':
                    completed_names.append('Psychologie & Gesellschaft')
            
            if completed_names:
                prompt_parts.append(f"✅ Bereits abgeschlossen: {', '.join(completed_names)}")
        
        prompt_parts.append("Frage: 'Was interessiert dich mehr?' oder 'Womit sollen wir weitermachen?'")
        prompt_parts.append("Sei enthusiastisch und mache Lust auf die verfügbaren Optionen.")
        
        return " ".join(prompt_parts)

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