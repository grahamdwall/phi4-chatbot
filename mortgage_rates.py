import requests
import datetime
import re
import googlemaps


def check_location_in_ontario(location_str, api_key):
    """
    Determines if the given location is in Ontario, Canada, elsewhere in Canada, or outside Canada.

    Parameters:
    - location_str (str): The location to check.
    - api_key (str): Your Google Maps Geocoding API key.

    Returns:
    - str: 'Ontario, Canada', 'Canada (outside Ontario)', 'Outside Canada', or 'Undetermined'.
    """
    gmaps = googlemaps.Client(key=api_key)

    try:
        # Geocode the location
        geocode_result = gmaps.geocode(location_str)

        if not geocode_result:
            return 'Undetermined'

        # Extract address components
        address_components = geocode_result[0].get('address_components', [])

        # Initialize variables
        country = None
        province = None

        # Parse address components
        for component in address_components:
            if 'country' in component['types']:
                country = component['long_name']
            if 'administrative_area_level_1' in component['types']:
                province = component['long_name']

        # Determine location
        if country == 'Canada':
            if province == 'Ontario':
                return 'Ontario, Canada'
            elif province:
                return 'Canada (outside Ontario)'
            else:
                return 'Canada (province undetermined)'
        elif country:
            return 'Outside Canada'
        else:
            return 'Undetermined'

    except Exception as e:
        print(f"Error: {e}")
        return 'Undetermined'

def calculate_min_downpayment(home_price):
    """
    Calculate the minimum down payment required for a given home price in Canada.

    Parameters:
    home_price (float): The purchase price of the home.

    Returns:
    float: The minimum down payment required.
    """
    if home_price <= 500000:
        return home_price * 0.05
    elif home_price < 1500000:
        return 500000 * 0.05 + (home_price - 500000) * 0.10
    else:
        return home_price * 0.20

def calculate_cmhc_insurance(purchase_price: float, down_payment: float) -> float:
    """
    Calculate the CMHC mortgage default insurance premium.

    :param purchase_price: The total purchase price of the home.
    :param down_payment: The amount of down payment made.
    :return: The CMHC insurance premium amount.
    """
    min_down_payment = calculate_min_downpayment(purchase_price)
    # Check if the down payment is less than the minimum required
    if down_payment < min_down_payment:
        down_payment = min_down_payment # clamp down payment to minimum required to ensure calculation can proceed
        #raise ValueError(f"Down payment must be at least {min_down_payment:.2f} for a purchase price of {purchase_price:.2f}")

    # Check if the down payment is greater than the purchase price
    if down_payment > purchase_price:
        down_payment = purchase_price # clamp down payment to purchase price to ensure calculation can proceed
        #raise ValueError("Down payment cannot exceed the purchase price")

    # Check if the purchase price is less than or equal to zero
    if purchase_price <= 0:
        purchase_price = 1 # clamp purchase price to 1 to ensure calculation can proceed
        #raise ValueError("Purchase price must be greater than zero")
    
    # Check if the down payment is less than or equal to zero
    if down_payment <= 0:
        down_payment = 1 # clamp down payment to 1 to ensure calculation can proceed
        #raise ValueError("Down payment must be greater than zero")
    
    # CMHC insurance is not available for homes priced at $1,500,000 or more
    if purchase_price >= 1_500_000:
        return 0.0  # Insurance not applicable

    # Calculate the loan amount
    loan_amount = purchase_price - down_payment

    # Calculate the Loan-to-Value (LTV) ratio
    ltv = loan_amount / purchase_price

    # Determine the premium rate based on LTV
    if ltv > 0.90:
        premium_rate = 0.04  # 4.00%
    elif ltv > 0.85:
        premium_rate = 0.031  # 3.10%
    elif ltv > 0.80:
        premium_rate = 0.028  # 2.80%
    elif ltv > 0.75:
        premium_rate = 0.024  # 2.40%
    elif ltv > 0.65:
        premium_rate = 0.017  # 1.70%
    else:
        premium_rate = 0.006  # 0.60%

    # Calculate the insurance premium
    insurance_premium = loan_amount * premium_rate
    return round(insurance_premium, 2)

def fetch_latest_mortgage_rates():
    interest_rate_data = {}

    # Calculate date range: last 12 months
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)

    # Format dates as strings
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    # Construct the API URL
    url = f'https://www.bankofcanada.ca/valet/observations/group/A4_RATES_MORTGAGES/json?start_date={start_date_str}&end_date={end_date_str}&order_dir=desc'

    # Make the GET request
    response = requests.get(url, headers={'accept': 'application/json'})

    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return

    data = response.json()
    series_detail = data.get('seriesDetail', {})
    observations = data.get('observations', [])

    if not observations:
        print("No observations found.")
        return

    # Get the most recent observation
    latest_observation = observations[0]

    for key, value in latest_observation.items():
        if key == 'd':
            continue  # Skip the date field

        rate_info = value.get('v')
        if rate_info is None:
            continue  # Skip if rate is missing

        series_info = series_detail.get(key, {})
        description = series_info.get('description', 'Unknown')
        label = series_info.get('label', 'Unknown')

        # Determine Insured
        insured = True
        if 'uninsured' in description:
            insured = False
        elif 'insured' in description:
            insured = True
        else:
            continue  # Skip if insured is unknown

        # Determine Down Payment
        if not 'Funds advanced' in description:
            continue  # keep only the Funds advanced rates for calculating cost of new mortgage (Outstanding balances is like a historical average)

        # Determine Type
        if 'Variable rate' in label:
            rate_type = 'Variable'
        elif 'Fixed rate' in label:
            rate_type = 'Fixed'
        else:
            continue  # Skip if type is unknown

        # Extract Term
        term_match = re.search(r'Fixed rate, (.+)', label)
        if term_match:
            term = term_match.group(1).strip()
        else:
            continue  # Skip if term is unknown or variable

        # Parse term into from and to
        term_from = ''
        term_to = ''
        if term.startswith('<'):
            match = re.search(r'^\D*(\d+)', term)
            if match:
                first_number = match.group(1)
            term_from = '0'
            term_to = first_number
        elif 'to' in term:
            parts = term.split('to')
            match = re.search(r'^\D*(\d+)', parts[0])
            if match:
                first_number = match.group(1)
            match = re.search(r'^\D*(\d+)', parts[1])
            if match:
                second_number = match.group(1)
            term_from = first_number
            term_to = second_number
        elif 'and over' in term:
            match = re.search(r'^\D*(\d+)', term)
            if match:
                first_number = match.group(1)
            term_from = first_number
            term_to = '30'
        else:
            continue  # Skip if term format is unrecognized

        #print(f"key: {key} Term From: {term_from}, Term To: {term_to}, Insured: {insured}, Rate: {rate_info}%")
        interest_rate_data[key] = {
            'term_from': term_from,
            'term_to': term_to,
            'insured': insured,
            'rate': rate_info
        }
    return interest_rate_data

def get_interest_rate_percent(term_years: float, insured: bool) -> float:
    """
    Returns the interest rate for a given term length and insurance status.
    """
    print("get_interest_rates()")
    print("term_years:", term_years)
    print("insured:", insured)
    rates = fetch_latest_mortgage_rates()
    if not rates:
        print("No rate data available.")
        return None

    print(rates)

    # Filter rates based on term and insurance status
    matching_rates = []
    for key in rates:
        rate = rates[key]
        #print(rate)
        if rate['insured'] == insured and rate['term_from'] is not None and rate['term_to'] is not None and int(rate['term_from']) < term_years <= int(rate['term_to']):
            matching_rates.append(rate)
            break

    if not matching_rates:
        print("No matching rate found.")
        return None

    # If multiple rates match, return the one with the lowest rate
    best_rate = min(matching_rates, key=lambda x: float(x['rate']))
    return best_rate['rate']

# Example usage:
if __name__ == "__main__":
    term_length = 5  # in years
    is_insured = True
    rate = get_interest_rate_percent(term_length, is_insured)
    if rate is not None:
        print(f"The interest rate for a {term_length}-year {'insured' if is_insured else 'uninsured'} mortgage is {rate}%.")
    else:
        print("Interest rate not found for the given criteria.")
