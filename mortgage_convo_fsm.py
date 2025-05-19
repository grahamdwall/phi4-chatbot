from transitions import Machine
from mortgage_rates import check_location_in_ontario
from dotenv import load_dotenv
import os

class MortgageConversation:
    required_fields = ['price', 'location']
    optional_fields = ['first_time_home_buyer', 'down_payment', 'term_years', 'payment_frequency']

    def __init__(self):
        self.collected_data = {}
        self.state = 'collecting_price'
        self.machine = Machine(model=self, states=['collecting_price', 'estimate', 'show_listings', 'book_meeting', 'error'], initial='collecting_price')

        # Transitions based on input data availability
        self.machine.add_transition(trigger='update', source='collecting_location', dest='collecting_location', conditions=['location_invalid'])
        self.machine.add_transition(trigger='update', source='collecting_location', dest='book_meeting', conditions=['location_outside_ontario'])
        self.machine.add_transition(trigger='update', source='collecting_location', dest='collecting_price', conditions=['location_ontario'])
        self.machine.add_transition(trigger='update', source='collecting_price', dest='estimate', conditions=['price_valid'])
        self.machine.add_transition(trigger='reset', source='*', dest='collecting', after='clear_data')

    def price_valid(self):
        if self.collected_data.get('price') <= 0:
            print("Error: Price is not valid.")
            return False
        return True

    def _check_location_category(self):
        load_dotenv("/workspace/.env")
        token = os.getenv("GOOGLE_MAPS_API_KEY")
        if not token or 'location' not in self.collected_data:
            return "invalid"

        result = check_location_in_ontario(self.collected_data['location'], token)

        if result in ['Ontario, Canada']:
            return "ontario"
        elif result in ['Canada (outside Ontario)', 'Outside Canada']:
            return "outside"
        else:
            return "invalid"
    
    def location_ontario(self):
        return self._check_location_category() == "ontario"

    def location_outside_ontario(self):
        return self._check_location_category() == "outside"

    def location_invalid(self):
        return self._check_location_category() == "invalid"

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