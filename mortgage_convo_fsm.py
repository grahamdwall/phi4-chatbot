from transitions import Machine

class MortgageConversation:
    required_fields = ['principal', 'interest_rate', 'term_years', 'payment_frequency']

    def __init__(self):
        self.collected_data = {}
        self.state = 'collecting'
        self.machine = Machine(model=self, states=['collecting', 'ready', 'error'], initial='collecting')

        # Transitions based on input data availability
        self.machine.add_transition(trigger='update', source='collecting', dest='ready', conditions=['is_complete'])
        self.machine.add_transition(trigger='update', source='collecting', dest='collecting')
        self.machine.add_transition(trigger='reset', source='*', dest='collecting', after='clear_data')

    def is_complete(self):
        return all(field in self.collected_data for field in self.required_fields)

    def receive_input(self, field, value):
        self.collected_data[field] = value
        print(f"Received: {field} = {value}")
        self.update()

    def clear_data(self):
        self.collected_data = {}
        print("Conversation reset.")

    def missing_fields(self) -> list:
        """Returns a list of required fields that have not yet been provided."""
        return [field for field in self.required_fields if field not in self.collected_data]

# --- Example Usage ---
#chat = MortgageConversation()

#chat.receive_input('principal', 300000)
#chat.receive_input('interest_rate', 4.2)
#chat.receive_input('term_years', 30)
#chat.receive_input('payment_frequency', 'monthly')

#print("State:", chat.state)
#print("Collected:", chat.collected_data)