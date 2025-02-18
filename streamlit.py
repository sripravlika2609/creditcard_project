import streamlit as st
import re

def luhn_algorithm(card_number):
    digits = [int(d) for d in card_number][::-1]
    checksum = 0
    
    for i, digit in enumerate(digits):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    
    return checksum % 10 == 0

def get_card_type(card_number):
    card_patterns = {
        "Visa": r"^4[0-9]{12}(?:[0-9]{3})?$",
        "MasterCard": r"^5[1-5][0-9]{14}$|^2[2-7][0-9]{14}$",
        "American Express": r"^3[47][0-9]{13}$",
        "Discover": r"^6(?:011|5[0-9]{2})[0-9]{12}$"
    }
    
    for card_type, pattern in card_patterns.items():
        if re.match(pattern, card_number):
            return card_type
    
    return "Unknown"

st.title("Credit Card Validator")

card_number = st.text_input("Enter Credit Card Number", max_chars=19)

if st.button("Validate"):
    card_number = card_number.replace(" ", "")
    if not card_number.isdigit():
        st.error("Invalid input! Please enter only numeric values.")
    else:
        is_valid = luhn_algorithm(card_number)
        card_type = get_card_type(card_number)
        
        if is_valid:
            st.success(f"Valid {card_type} Card")
        else:
            st.error("Invalid Card Number")
