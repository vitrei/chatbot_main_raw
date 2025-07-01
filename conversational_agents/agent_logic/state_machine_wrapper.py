from transitions import Machine
import json
import os
from .generic_state_machine import GenericStateMachine, load_state_machine_config as generic_load_config

class ConversationStateMachine(GenericStateMachine):
    """Conversation-specific state machine wrapper using generic backend"""
    
    def __init__(self, config):
        """Initialize conversation state machine with generic backend"""
        
        # Extract states from current stage if available
        current_stage = config.get('current_stage', 'onboarding')
        stages = config.get('stages', {})
        
        if current_stage in stages and 'states' in stages[current_stage]:
            config['states'] = stages[current_stage]['states']
        else:
            # Fallback: extract all unique states from transitions
            all_states = set()
            for transition in config.get('transitions', []):
                all_states.add(transition['dest'])
                source = transition['source']
                if isinstance(source, list):
                    all_states.update(source)
                elif source != '*':
                    all_states.add(source)
            config['states'] = list(all_states)
        
        # Initialize generic state machine
        super().__init__(config)
        
        # Legacy properties for backward compatibility
        self.turn_counter = 0
        
        # Conversation-specific setup
        self.set_fake_news_stimulus_url(None)
    
    def set_fake_news_stimulus_available(self, user_id):
        """Set fake news stimulus as available with URL"""
        self.fake_news_stimulus_url = f"http://localhost:8000/instagram_tablet/{user_id}"
        print(f"🎬 FAKE NEWS STIMULUS AVAILABLE: {self.fake_news_stimulus_url}")

    def set_fake_news_stimulus_url(self, url):
        """Set fake news stimulus URL"""
        self.fake_news_stimulus_url = url

    def is_fake_news_stimulus_available(self):
        """Check if fake news stimulus is available"""
        return hasattr(self, 'fake_news_stimulus_url') and self.fake_news_stimulus_url is not None
    
    def get_state_context_for_decision_agent(self, turn_counter: int = 0):
        """Provide enhanced state context for decision agent"""
        # Update internal turn counter
        self.turn_counter = turn_counter
        
        # Get generic context
        context = super().get_state_context_for_decision_agent(turn_counter)
        
        # Add conversation-specific context
        context.update({
            'fake_news_available': self.is_fake_news_stimulus_available(),
            'fake_news_stimulus_url': getattr(self, 'fake_news_stimulus_url', None),
            'stage_progress': self.get_stage_progress(turn_counter)
        })
        
        return context
    
    def get_stage_progress(self, turn_counter):
        """Get progress information for current stage"""
        stage_info = self.stages.get(self.current_stage, {})
        target_turns = stage_info.get('target_turns', 20)
        milestones = stage_info.get('progression_milestones', {})
        
        # Check milestone status
        milestone_status = []
        for milestone_turn, milestone_desc in milestones.items():
            milestone_turn_num = int(milestone_turn.replace('turn_', ''))
            if turn_counter >= milestone_turn_num:
                milestone_status.append(f"{milestone_desc}: due")
            else:
                milestone_status.append(f"{milestone_desc}: upcoming")
        
        progress_percentage = min(100, (turn_counter / target_turns) * 100)
        
        return {
            'progress_percentage': progress_percentage,
            'milestone_status': milestone_status,
            'target_turns': target_turns
        }

# Cache for state machine config to avoid repeated file reads
_state_machine_config_cache = None

def load_state_machine_config():
    """Load state machine configuration from separate JSON file (with caching)"""
    global _state_machine_config_cache
    
    if _state_machine_config_cache is not None:
        return _state_machine_config_cache
    
    # Use generic loader
    config = generic_load_config()
    if config:
        _state_machine_config_cache = config
    
    return config

def create_state_machine_from_prompts(prompts):
    """Create state machine from separate state_machine.json file"""
    try:
        config = load_state_machine_config()
        if not config:
            print("❌ No state machine config loaded")
            return None
        
        return ConversationStateMachine(config)
    except Exception as e:
        print(f"❌ Error creating state machine: {e}")
        return None