from transitions.extensions import GraphMachine

class DeepfakeAwarenessFlow:
    states = [
        'init_greeting',      # Initiale Begrüßung
        'engagement_hook',    # User engagieren
        'stimulus_present',   # Stimulus/Deepfake präsentieren
        'reaction_wait',      # Auf User-Reaktion warten
        'explore_path',       # User glaubt - Pfad erkunden
        'confirm_skepticism', # User ist skeptisch - bestätigen
        'comfort_user',       # User ist aufgebracht - beruhigen
        'meta_reflection',    # User ist distanziert - Meta-Reflexion
        'critical_thinking',  # Kritisches Denken fördern
        'learning_transition',# Lernübergang
        'conversation_closure',# Gespräch beenden
        'conversation_repair' # Gesprächsreparatur
    ]
    
    def __init__(self):
        self.context = {}
        self.machine = GraphMachine(
            model=self,
            states=DeepfakeAwarenessFlow.states,
            initial='init_greeting',
            auto_transitions=False,
            show_conditions=True,
            title="Deepfake Awareness Flow",
            show_state_attributes=True
        )
        
        # 🎯 Haupt-Flow Transitions
        self.machine.add_transition('proceed_normally', 'init_greeting', 'engagement_hook')
        self.machine.add_transition('engage_user', 'engagement_hook', 'stimulus_present')
        self.machine.add_transition('present_stimulus', 'stimulus_present', 'reaction_wait')
        
        # 🔄 Reaktions-Handling nach stimulus_present
        self.machine.add_transition('user_believes', 'reaction_wait', 'explore_path')
        self.machine.add_transition('user_skeptical', 'reaction_wait', 'confirm_skepticism')
        self.machine.add_transition('user_upset', 'reaction_wait', 'comfort_user')
        self.machine.add_transition('user_detached', 'reaction_wait', 'meta_reflection')
        
        # 🧠 Übergang zum kritischen Denken von verschiedenen Zuständen
        self.machine.add_transition('move_to_thinking', 
                                   source=['explore_path', 'confirm_skepticism', 
                                          'comfort_user', 'meta_reflection'], 
                                   dest='critical_thinking')
        
        # 📚 Lern- und Abschluss-Flow
        self.machine.add_transition('ready_to_learn', 'critical_thinking', 'learning_transition')
        self.machine.add_transition('finish_conversation', 'learning_transition', 'conversation_closure')
        
        # 🔧 Reparatur-Mechanismen (strategisch begrenzt - nicht von init/closure)
        self.machine.add_transition('need_repair', 
                                   source=['engagement_hook', 'stimulus_present', 'reaction_wait', 
                                          'explore_path', 'confirm_skepticism', 'comfort_user',
                                          'meta_reflection', 'critical_thinking', 'learning_transition'],
                                   dest='conversation_repair')
        
        # 🔄 Aus Reparatur heraus
        self.machine.add_transition('restart_engagement', 'conversation_repair', 'engagement_hook')
        self.machine.add_transition('force_closure', 'conversation_repair', 'conversation_closure')
        
        # 🔄 Re-engagement Optionen
        self.machine.add_transition('re_engage', 
                                   source=['stimulus_present', 'critical_thinking'], 
                                   dest='engagement_hook')
        
        # 🤗 Comfort nur von emotional relevanten Zuständen
        self.machine.add_transition('comfort_needed',
                                   source=['reaction_wait', 'explore_path', 'confirm_skepticism', 
                                          'critical_thinking', 'conversation_repair'],
                                   dest='comfort_user')
    
    def available_transitions(self):
        """Gibt verfügbare Übergänge vom aktuellen Zustand zurück"""
        return self.machine.get_triggers(self.state)
    
    def is_complete(self):
        """Prüft ob das Gespräch abgeschlossen ist"""
        return self.state == 'conversation_closure'
    
    def save_context(self, key, value):
        """Speichert Kontext-Information"""
        self.context[key] = value
    
    def get_context(self, key=None):
        """Holt Kontext-Information"""
        if key:
            return self.context.get(key)
        return self.context
    
    def get_current_state(self):
        """Gibt den aktuellen Zustand zurück"""
        return self.state
    
    def get_state_description(self):
        """Gibt eine Beschreibung des aktuellen Zustands zurück"""
        descriptions = {
            'init_greeting': 'Initiale Begrüßung - Gespräch beginnt',
            'engagement_hook': 'User wird engagiert und vorbereitet',
            'stimulus_present': 'Deepfake/Stimulus wird präsentiert',
            'reaction_wait': 'Warten auf User-Reaktion zum Stimulus',
            'explore_path': 'User glaubt - Pfad wird erkundet',
            'confirm_skepticism': 'User ist skeptisch - Skepsis wird bestätigt',
            'comfort_user': 'User ist aufgebracht - wird beruhigt',
            'meta_reflection': 'User ist distanziert - Meta-Reflexion',
            'critical_thinking': 'Kritisches Denken wird gefördert',
            'learning_transition': 'Übergang zum Lernen',
            'conversation_closure': 'Gespräch wird beendet',
            'conversation_repair': 'Gesprächsreparatur aktiv'
        }
        return descriptions.get(self.state, f"Unbekannter Zustand: {self.state}")
    
    def draw(self, filename="deepfake_awareness_flow.png"):
        """Erstellt eine visuelle Darstellung der State Machine"""
        try:
            graph = self.machine.get_graph()
            graph.draw(filename, prog="dot")
            print(f"State Machine Diagramm gespeichert als {filename}")
        except Exception as e:
            print(f"Fehler beim Erstellen des Diagramms: {e}")
            print("Stelle sicher, dass graphviz installiert ist: pip install graphviz")

# 🧪 Beispiel-Nutzung
if __name__ == "__main__":
    # State Machine erstellen
    flow = DeepfakeAwarenessFlow()
    
    print(f"Startzustand: {flow.get_current_state()}")
    print(f"Beschreibung: {flow.get_state_description()}")
    print(f"Verfügbare Übergänge: {flow.available_transitions()}")
    
    # Beispiel-Flow durchlaufen
    print("\n--- Beispiel-Durchlauf ---")
    
    # Normaler Start
    flow.proceed_normally()
    print(f"Nach proceed_normally: {flow.get_current_state()}")
    
    # User engagieren
    flow.engage_user()
    print(f"Nach engage_user: {flow.get_current_state()}")
    
    # Stimulus präsentieren
    flow.present_stimulus()
    print(f"Nach present_stimulus: {flow.get_current_state()}")
    
    # User glaubt dem Fake
    flow.user_believes()
    flow.save_context('user_reaction', 'believed_fake')
    print(f"Nach user_believes: {flow.get_current_state()}")
    
    # Zum kritischen Denken überleiten
    flow.move_to_thinking()
    print(f"Nach move_to_thinking: {flow.get_current_state()}")
    
    # Bereit zum Lernen
    flow.ready_to_learn()
    print(f"Nach ready_to_learn: {flow.get_current_state()}")
    
    # Gespräch beenden
    flow.finish_conversation()
    print(f"Gespräch beendet: {flow.get_current_state()}")
    print(f"Ist abgeschlossen: {flow.is_complete()}")
    
    print(f"\nGespeicherter Kontext: {flow.get_context()}")
    
    # Diagramm erstellen (optional)
    flow.draw()