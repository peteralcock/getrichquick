import socket
import whois
import requests
from datetime import datetime, timedelta

# Define expiration threshold (in days)
EXPIRATION_THRESHOLD_DAYS = 30

def check_domain_availability(domain):
    """Check if the domain has a DNS entry. Returns True if available."""
    try:
        # Try to resolve the domain. If it succeeds, it's registered.
        socket.gethostbyname(domain)
        return False
    except socket.gaierror:
        # If resolving fails, the domain may be available.
        return True

def check_whois_expiration(domain):
    """Check WHOIS expiration date to see if the domain is expiring soon."""
    try:
        domain_info = whois.whois(domain)
        expiration_date = domain_info.expiration_date
        # Handle cases where expiration_date might be a list
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        if expiration_date:
            # Calculate days until expiration
            days_until_expiration = (expiration_date - datetime.now()).days
            if days_until_expiration <= EXPIRATION_THRESHOLD_DAYS:
                return True, expiration_date
        return False, expiration_date
    except Exception as e:
        print(f"Error checking WHOIS for {domain}: {e}")
        return False, None

def check_http_status(domain):
    """Check if an HTTP request to the domain succeeds, indicating activity."""
    try:
        response = requests.get(f"http://{domain}", timeout=3)
        # If request succeeds, assume the domain is in use
        return response.status_code == 200
    except requests.RequestException:
        return False

def main():
    input_file = "domains.txt"        # Input file with domain names (one per line)
    available_file = "available.txt"   # Output file for available domains
    comingsoon_file = "comingsoon.txt" # Output file for soon-to-expire domains

    # Read domain names from file
    with open(input_file, "r") as file:
        domains = [line.strip() for line in file if line.strip()]

    # Lists to store results
    available_domains = []
    comingsoon_domains = []

    # Check each domain for availability and expiration
    for domain in domains:
        if check_domain_availability(domain):
            available_domains.append(domain)
            print(f"{domain} is available")
        else:
            expiring_soon, expiration_date = check_whois_expiration(domain)
            if expiring_soon:
                comingsoon_domains.append(f"{domain} - expires on {expiration_date}")
                print(f"{domain} is expiring soon on {expiration_date}")
            elif check_http_status(domain):
                print(f"{domain} is active and taken")

    # Write available domains to output file
    with open(available_file, "w") as file:
        for domain in available_domains:
            file.write(domain + "\n")

    # Write soon-to-expire domains to output file
    with open(comingsoon_file, "w") as file:
        for entry in comingsoon_domains:
            file.write(entry + "\n")

    print(f"\nAvailable domains saved to {available_file}")
    print(f"Domains expiring soon saved to {comingsoon_file}")

if __name__ == "__main__":
    main()
