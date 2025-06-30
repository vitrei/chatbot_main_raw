from transitions import Machine
import json

class ConversationStateMachine:
    def __init__(self, config):
        """
        Initialize state machine from configuration - manages conversation flow only
        """
        self.states = config['states']
        self.transitions_config = config['transitions']
        self.current_state = config['initial_state']
        
        # Create the machine
        self.machine = Machine(
            model=self,
            states=self.states,
            initial=config['initial_state'],
            transitions=self.transitions_config
        )
        
        print(f"🎰 STATE MACHINE INITIALIZED: {self.current_state}")

    def get_current_state(self):
        """Get current conversation state"""
        return self.state

    def get_available_transitions(self):
        """Get available state transitions from current state"""
        available_triggers = []
        
        for transition in self.transitions_config:
            # Check if current state is in source (handle both string and list sources)
            source = transition['source']
            if source == '*' or source == self.state or (isinstance(source, list) and self.state in source):
                available_triggers.append({
                    'trigger': transition['trigger'],
                    'dest': transition['dest'],
                    'description': self.get_trigger_description(transition['trigger'])
                })
        
        return available_triggers

    def get_trigger_description(self, trigger):
        """Get human-readable description for trigger"""
        descriptions = {
            'proceed_normally': 'Normal progression forward',
            'engage_user': 'User shows interest, continue engagement',
            'present_stimulus': 'User ready for stimulus presentation', 
            'user_believes': 'User believes/accepts the stimulus',
            'user_skeptical': 'User is skeptical about stimulus',
            'user_upset': 'User is emotionally upset/confused',
            'user_detached': 'User is ironical/detached',
            'move_to_thinking': 'Progress to critical thinking phase',
            'ready_to_learn': 'User ready for learning transition',
            'finish_conversation': 'End conversation successfully',
            'need_repair': 'Conversation needs repair/reset',
            'restart_engagement': 'Restart engagement from repair',
            'force_closure': 'Force end conversation',
            're_engage': 'Re-engage user (go back)',
            'comfort_needed': 'User needs emotional comfort'
        }
        return descriptions.get(trigger, 'Unknown trigger')

    def can_trigger(self, trigger_name):
        """Check if a trigger can be executed from current state"""
        try:
            available = [t['trigger'] for t in self.get_available_transitions()]
            return trigger_name in available
        except:
            return False

    def execute_transition(self, trigger_name, reason="No reason provided"):
        """Execute a state transition - called by Decision Agent if needed"""
        if not self.can_trigger(trigger_name):
            print(f"❌ INVALID STATE TRANSITION: {trigger_name} from {self.state}")
            return False
        
        old_state = self.state
        
        try:
            # Execute the trigger
            trigger_method = getattr(self, trigger_name)
            trigger_method()
            
            print(f"✅ STATE MACHINE TRANSITION: {old_state} --{trigger_name}--> {self.state}")
            print(f"📝 TRANSITION REASON: {reason}")
            return True
            
        except Exception as e:
            print(f"❌ STATE TRANSITION FAILED: {trigger_name} - {str(e)}")
            return False

    def get_state_context_for_decision_agent(self):
        """Provide state context for decision agent"""
        return {
            'current_state': self.state,
            'available_transitions': self.get_available_transitions(),
            'state_description': f"Conversation is in phase: {self.state}",
            'fake_news_stimulus_url': getattr(self, 'fake_news_stimulus_url', None)
        }

    def set_fake_news_stimulus_available(self, user_id):
        """Set fake news stimulus as available with URL"""
        self.fake_news_stimulus_url = f"http://localhost:8000/instagram_tablet/{user_id}"
        print(f"🎬 FAKE NEWS STIMULUS AVAILABLE: {self.fake_news_stimulus_url}")

    def is_fake_news_stimulus_available(self):
        """Check if fake news stimulus is available"""
        return hasattr(self, 'fake_news_stimulus_url') and self.fake_news_stimulus_url is not None

def create_state_machine_from_prompts(prompts):
    """Create state machine from prompts configuration"""
    try:
        config = prompts.get('state_machine_config', {})
        if not config:
            print("❌ No state machine config found in prompts")
            return None
        
        return ConversationStateMachine(config)
    except Exception as e:
        print(f"❌ Error creating state machine: {e}")
        return None