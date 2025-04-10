#encrypt.py
import random
import secrets
import math
#import json
import string

#Cuts a text like a deck of cards and swaps halves
def cut_deck(text):
    middle = len(text) // 2

    if len(text) % 2 == 0:
        return text[middle:] + text[:middle]
    else:
        return text[middle+1:] + text[middle] + text[:middle]

#Shuffle a text by a certain step
def shuffle(text, step):
    length = len(text)
    result = []

    for i in range(0, length, step):
        result.append(text[i])

    remaining_chars = [text[i] for i in range(length) if i % step != 0]

    result.extend(remaining_chars)

    return ''.join(result)

#Reverse cases up->low and low->up
def reverse_case(text):

      return ''.join([char.lower() if char.isupper() else char.upper() for char in text])

#Insert a char at a specific index
def insert_char_at_index(text, char, index):
    if index < 0 or index > len(text):
        raise ValueError("Index out of range")
    
    return text[:index] + char + text[index:]

#Replace chars with the shifted ones in the dictionary
def shift(text, char_map):
    result = []
    char_map_len = len(char_map)
    
    for idx, char in enumerate(text):
        if char in char_map:
            index = char_map.index(char)
            shift_amount = idx  # Step is the index of char in text
            
            if idx % 2 == 0:
                shifted_index = (index + shift_amount) % char_map_len  # Shift to the right
            else:
                shifted_index = (index - shift_amount) % char_map_len  # Shift to the left
            
            shifted_char = char_map[shifted_index]
            result.append(shifted_char)
        else:
            result.append(char)
    
    return ''.join(result)

#Digital root
def digital_root(n):
    while n >= 10:
        n = sum(int(digit) for digit in str(n))
    return n

#Random rule for encryption
def random_rules():
    return chr(random.randint(32, 126))


def custom_encrypt_l1(text):
    
    #Initialize values
    encrypted_text=text
    dictionary = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    
    dictionary_char = random_rules()
    string_char = random_rules()
    control_char = random_rules()

    encrypt_factors=[2,3,5,7]
    dictionary_rules=bin(ord(dictionary_char))[2:].zfill(8)[:4]
    string_rules=bin(ord(string_char))[2:].zfill(8)[:4]
    control_rules=bin(ord(control_char))[2:].zfill(8)[:4]

    #Encrypt
    for factor, d_rule, s_rule, c_rule in zip(encrypt_factors, dictionary_rules, string_rules, control_rules):
        
        dictionary=shuffle(dictionary, factor)
        encrypted_text=shuffle(encrypted_text, factor)

        if d_rule == '1':
            dictionary = reverse_case(dictionary)
        else:
            dictionary = cut_deck(dictionary)

        if s_rule == '1':
            encrypted_text = reverse_case(encrypted_text)
        else:
            encrypted_text = cut_deck(encrypted_text)

        if c_rule == '1':
            encrypted_text = shift(encrypted_text, dictionary)

        encrypted_text = shift(encrypted_text, dictionary)

    encrypted_text = insert_char_at_index(encrypted_text, control_char, digital_root(len(encrypted_text)))
    encrypted_text = insert_char_at_index(encrypted_text, string_char, 0)
    encrypted_text = insert_char_at_index(encrypted_text, dictionary_char, len(encrypted_text))

    return encrypted_text


#Custom encryption layer 2

def fold_on_primary_diagonal(matrix):
    """
    Transposes the given square matrix by swapping elements along its primary diagonal.

    Parameters:
    matrix (list of list of int/float): A square matrix represented as a list of lists.

    Returns:
    list of list of int/float: The matrix transposed along its primary diagonal.

    Example:
    Input: [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

    Output: [[1, 4, 7],
             [2, 5, 8],
             [3, 6, 9]]
    """
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix

def fold_on_secondary_diagonal(matrix):
    """
    Swaps the elements of the given square matrix to effectively fold it along the secondary diagonal.

    Parameters:
    matrix (list of list of int/float): A square matrix represented as a list of lists.

    Returns:
    list of list of int/float: The matrix mirrored along the secondary diagonal.

    Example:
    Input: [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

    Output: [[3, 2, 1],
             [6, 5, 4],
             [9, 8, 7]]
    """
    n = len(matrix)
    for i in range(n):
        for j in range(n // 2):
            opposite_j = n - 1 - j
            matrix[i][j], matrix[i][opposite_j] = matrix[i][opposite_j], matrix[i][j]
    return matrix

def fold_horizontal(matrix):
    """
    Reflects the given matrix across the horizontal middle axis, effectively flipping it vertically.

    Parameters:
    matrix (list of list of int/float): A matrix represented as a list of lists.

    Returns:
    list of list of int/float: The matrix flipped along the horizontal axis.

    Example:
    Input: [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

    Output: [[7, 8, 9],
             [4, 5, 6],
             [1, 2, 3]]
    """
    n = len(matrix)
    for i in range(n // 2):
        matrix[i], matrix[n - 1 - i] = matrix[n - 1 - i], matrix[i]
    return matrix

def fold_vertical(matrix):
    """
    Reflects the given matrix across the vertical middle axis, effectively flipping it horizontally.

    Parameters:
    matrix (list of list of int/float): A matrix represented as a list of lists.

    Returns:
    list of list of int/float: The matrix flipped along the vertical axis.

    Example:
    Input: [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

    Output: [[3, 2, 1],
             [6, 5, 4],
             [9, 8, 7]]
    """
    n = len(matrix)
    m = len(matrix[0])
    for i in range(n):
        for j in range(m // 2):
            matrix[i][j], matrix[i][m - 1 - j] = matrix[i][m - 1 - j], matrix[i][j]
    return matrix

def string_to_matrix_dynamic(text):
    """
    Converts a string into a square matrix suitable for encryption.
    The matrix size dynamically adjusts to fit the string length while adhering to a minimum and maximum size.

    Parameters:
        text (str): The input string to be converted into a matrix.

    Returns:
        list[list[str]]: A 2D list (matrix) containing the characters of the string, with padding if necessary.

    Raises:
        ValueError: If the string length exceeds the capacity of the maximum matrix size (256x256).
    """

    # Save the original string length
    original_length = len(text)
    
    # Minimum and maximum allowable matrix sizes
    min_size = 4  # Minimum matrix size is 4x4
    max_size = 256  # Maximum matrix size is 256x256
    
    # Calculate the smallest square matrix that can fit all characters of the string
    matrix_size = max(min_size, math.ceil(math.sqrt(original_length)))
    
    # Ensure the matrix size does not exceed the defined maximum
    if matrix_size > max_size:
        raise ValueError("Input string is too long to fit in a 256x256 matrix")
    
    # Total number of cells in the square matrix
    total_cells = matrix_size ** 2
    
    # Convert the input string into a list of characters
    chars = list(text)
    
    # If the string does not fill the matrix, add random padding characters
    # Random characters are chosen from printable ASCII characters, excluding control characters
    while len(chars) < total_cells:
        random_char = random.choice(string.printable[:-6])  # Exclude non-printable characters
        chars.append(random_char)
    
    # Build the matrix row by row
    matrix = []
    for i in range(matrix_size):
        # Slice the appropriate section of the character list for each row
        row = chars[i * matrix_size:(i + 1) * matrix_size]
        matrix.append(row)
    
    # Return the completed matrix
    return matrix

def matrix_to_string(matrix):
    """
    Converts a 2D matrix of characters into a single concatenated string.

    Parameters:
        matrix (list[list[str]]): A 2D list where each sub-list represents a row of characters.

    Returns:
        str: A single string obtained by concatenating all rows of the matrix.
    """
    # Flatten the matrix and concatenate all characters into a single string
    long_string = ''.join(''.join(row) for row in matrix)
    return long_string

def rotate_layer(matrix, layer, steps, direction):
    """
    Rotates a specified layer of a square matrix in place by a given number of steps
    either to the left or to the right. This version is optimized to avoid redundant rotations.

    Parameters:
        matrix (list[list[int]]): The square matrix to be rotated.
        layer (int): The index of the layer to rotate (0 for outermost layer).
        steps (int): The number of steps to rotate.
        direction (str): 'left' or 'right' to specify the direction of rotation.

    Returns:
        None: The function modifies the matrix in place.
    """
    n = len(matrix)
    if n == 0 or layer < 0 or layer >= (n + 1) // 2:
        return  # Invalid layer or empty matrix

    # Calculate boundaries for the current layer
    top, left, bottom, right = layer, layer, n - layer - 1, n - layer - 1
    
    # Extract elements from the current layer
    layer_elements = []

    # Top row
    layer_elements.extend(matrix[top][left:right + 1])
    # Right column (excluding the top element already added)
    layer_elements.extend(matrix[i][right] for i in range(top + 1, bottom + 1))
    # Bottom row (reversed, excluding the corner already added)
    layer_elements.extend(matrix[bottom][j] for j in range(right - 1, left - 1, -1))
    # Left column (reversed, excluding the corners already added)
    layer_elements.extend(matrix[i][left] for i in range(bottom - 1, top, -1))
    
    # Layer length
    layer_length = len(layer_elements)
    
    # Effective rotation steps using modulo
    steps %= layer_length
    if direction == 'right':
        steps = layer_length - steps  # Convert left rotation to equivalent right rotation
    
    # Perform the rotation
    rotated_elements = layer_elements[steps:] + layer_elements[:steps]
    
    # Put rotated elements back into the matrix
    idx = 0

    # Top row
    for j in range(left, right + 1):
        matrix[top][j] = rotated_elements[idx]
        idx += 1
    # Right column
    for i in range(top + 1, bottom + 1):
        matrix[i][right] = rotated_elements[idx]
        idx += 1
    # Bottom row
    for j in range(right - 1, left - 1, -1):
        matrix[bottom][j] = rotated_elements[idx]
        idx += 1
    # Left column
    for i in range(bottom - 1, top, -1):
        matrix[i][left] = rotated_elements[idx]
        idx += 1

def generate_character_with_bits(last_four_bits):
    """
    Generates a random printable character whose ASCII value ends with the specified 4 bits.

    Parameters:
        last_four_bits (int): An integer between 0 and 15 (inclusive) representing the last 4 bits
                              to embed into the generated character's ASCII value.

    Returns:
        str: A single printable character with the specified last 4 bits in its ASCII value.

    Raises:
        ValueError: If `last_four_bits` is not within the range 0 to 15.
    """

    # Ensure the input is within the valid range for 4 bits
    if not (0 <= last_four_bits < 16):
        raise ValueError("last_four_bits must be between 0 and 15")

    # Define the set of printable ASCII characters (excluding control characters)
    printable_chars = string.printable[:-6]  # Exclude non-printable control characters

    while True:
        # Generate a random printable character
        random_char = random.choice(printable_chars)

        # Extract the ASCII value of the character and clear its last 4 bits
        ascii_val = ord(random_char) & 0xF0  # Retain only the higher-order 4 bits (clear last 4 bits)

        # Combine the cleared ASCII value with the desired last 4 bits
        final_char = chr(ascii_val | last_four_bits)  # Use bitwise OR to set the last 4 bits

        # Check if the resulting character is still printable
        if final_char in printable_chars:
            return final_char

def number_to_chars(num):
    """
    Converts a non-negative integer into a sequence of characters by encoding its binary representation
    into groups of 4 bits, which are mapped to printable characters.

    Parameters:
        num (int): The non-negative integer to convert. Must be between 0 and 65535 inclusive.

    Returns:
        str: A string of characters representing the binary-encoded number.

    Raises:
        ValueError: If the input number is negative or exceeds 65535.
    """

    # Ensure the input is a non-negative integer
    if num < 0:
        raise ValueError("Number must be non-negative")

    # Determine the bit count needed to represent the number
    if num <= 15:  # Fits in 4 bits
        bit_count = 4
    elif num <= 255:  # Fits in 8 bits
        bit_count = 8
    elif num <= 4095:  # Fits in 12 bits
        bit_count = 12
    elif num <= 65535:  # Fits in 16 bits
        bit_count = 16
    else:
        raise ValueError("Number must be between 0 and 65535")

    # Divide the number's binary representation into 4-bit groups (stacks)
    # Reverse the order of the groups to match the original stack processing logic
    binary_string = format(num, f'0{bit_count}b')  # Convert number to binary with leading zeros
    stacks = [binary_string[i:i + 4] for i in range(0, bit_count, 4)][::-1]

    # Convert each 4-bit group into a printable character using generate_character_with_bits
    characters = ''.join(generate_character_with_bits(int(stack, 2)) for stack in stacks)

    # Return the final string of characters
    return characters

def chars_to_number(chars):
    """
    Converts a sequence of 1 to 4 characters back into the original integer 
    by decoding the last 4 bits of each character's ASCII value.

    Parameters:
        chars (str): A string containing 1 to 4 characters to be converted.

    Returns:
        int: The integer represented by the characters.

    Raises:
        ValueError: If the input string does not contain between 1 and 4 characters.
    """

    # Ensure the input string has a valid length
    if not 1 <= len(chars) <= 4:
        raise ValueError("Input must contain between 1 and 4 characters")

    # Convert each character into its binary representation (extracting the last 4 bits)
    bit_stacks = []
    for char in chars:
        # Extract the last 4 bits of the ASCII value
        char_bits = format(ord(char), '08b')[-4:]
        bit_stacks.append(char_bits)

    # Reverse the bit stacks to match the original encoding order
    bit_stacks = bit_stacks[::-1]

    # Combine the bits from all characters into a single binary string
    bit_string = ''.join(bit_stacks)

    # Convert the binary string back into an integer
    num = int(bit_string, 2)

    return num

def random_in_interval(start, end):
    return random.randint(start, end)

def custom_encrypt_l2(text):
    """
    Encrypts the given string by applying a series of matrix transformations and obfuscating metadata.

    Parameters:
        text (str): The string to encrypt.

    Returns:
        str: The encrypted string.
    """

    # Step 1: Save the original length and obfuscate it using number_to_chars
    original_length = len(text)
    obfuscated_length = number_to_chars(original_length)

    # Step 2: Generate a random number of steps between 2 and 2^8 - 1 (255) and obfuscate it
    random_steps = random_in_interval(2, 255)
    obfuscated_steps = number_to_chars(random_steps)

    # Step 3: Convert the string into a square matrix using the dynamic string_to_matrix_dynamic function
    matrix = string_to_matrix_dynamic(text)

    # Step 4: Apply transformations to each layer of the matrix
    n = len(matrix)
    for layer in range((n + 1) // 2):
        # Determine the direction of rotation for the current layer
        direction = 'left' if layer % 2 == 1 else 'right'

        # Rotate the current layer by the number of random steps
        rotate_layer(matrix, layer, random_steps, direction)

        # Apply folding operations based on whether the layer is odd or even
        if layer % 2 == 1:  # Odd layers
            matrix = fold_horizontal(matrix)
            matrix = fold_on_primary_diagonal(matrix)
        else:  # Even layers
            matrix = fold_vertical(matrix)
            matrix = fold_on_secondary_diagonal(matrix)

    # Step 5: Convert the modified matrix back to a string
    encrypted_string = matrix_to_string(matrix)

    # Step 6: Concatenate the obfuscated length, obfuscated steps, and the encrypted string
    result = obfuscated_steps + encrypted_string + obfuscated_length

    result = insert_char_at_index(result, str(len(obfuscated_length)), digital_root(len(result)))

    # Returning the encrypted result
    return result

def custom_decrypt_l2(text):
    """
    Decrypts the given string by reversing the series of matrix transformations and de-obfuscating metadata.

    Parameters:
        text (str): The encrypted string to decrypt.

    Returns:
        str: The decrypted string.
    """

    # Step 1: Extract the control character and determine the number of characters used to represent the original length.
    final_length = len(text)
    root = digital_root(final_length - 1)
    replace_position = root % final_length
    control_char = text[replace_position]

    # Determine how many characters were used to obfuscate the original length.
    # Extract the last 4 bits to determine the number of length characters (between 1 and 4)
    num_length_chars = (ord(control_char) & 0x0F) + 1  # Extract last 4 bits and add 1 (values 1-4)

    # Remove the control character from the input string.
    text = text[:replace_position] + text[replace_position + 1:]

    # Step 2: Extract the obfuscated length and steps.
    # Extract the obfuscated length from the end of the string using `num_length_chars`
    obfuscated_length = text[-num_length_chars:]
    original_length = chars_to_number(obfuscated_length)

    # Extract the obfuscated steps from the start of the string (4 characters used)
    obfuscated_steps = text[:4]
    random_steps = chars_to_number(obfuscated_steps)

    # Remove metadata from the encrypted string
    encrypted_string = text[4:-num_length_chars]

    # Step 3: Convert the remaining string back to a matrix
    matrix = string_to_matrix_dynamic(encrypted_string)

    # Step 4: Reverse transformations applied to each layer of the matrix
    n = len(matrix)
    for layer in reversed(range((n + 1) // 2)):
        # Determine the direction of rotation for the current layer (reverse of encryption)
        direction = 'right' if layer % 2 == 0 else 'left'

        # Reverse the fold operations based on whether the layer is even or odd
        if layer % 2 == 0:  # Even layers (reverse the original folding order)
            matrix = fold_vertical(matrix)
            matrix = fold_on_secondary_diagonal(matrix)
        else:  # Odd layers
            matrix = fold_horizontal(matrix)
            matrix = fold_on_primary_diagonal(matrix)

        # Rotate the current layer by the number of steps (opposite of encryption)
        rotate_layer(matrix, layer, random_steps, direction)



    # Step 5: Convert the modified matrix back to the original string
    decrypted_string = matrix_to_string(matrix)

    # Step 6: Truncate to the original length (remove any padding)
    decrypted_string = decrypted_string[:original_length]

    return decrypted_string


# nu merge decryptarea verific mai incolo
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

s = "ana are mere destule pentru toata iarna"
for i in range(1,100):

    print(custom_decrypt_l2(custom_encrypt_l2(s)))




# def custom_encrypt_l2(text):
#    text_length = len(text)
#    encryption_matrix = string_to_matrix(text)
#    nr_steps = 7

#    for i in range(1, len(encryption_matrix) // 2 + 1):
#        if i % 2 == 0:
#            rotate_layer(encryption_matrix, i, nr_steps, "left")
#            fold_vertical(encryption_matrix)
#            fold_on_primary_diagonal(encryption_matrix)
#        else:
#            rotate_layer(encryption_matrix, i, nr_steps, "right")
#            fold_horizontal(encryption_matrix)
#            fold_on_secondary_diagonal(encryption_matrix)
   
#    encrypted_string = matrix_to_string(encryption_matrix)
#    encrypted_len = number_to_chars(text_length)
#   encrypted_string += encrypted_len

#    return encrypted_string




    # Trebuie sa setez numarul de rotiri si spre ce parte este stanga sau dreatpta si sa vad unde pun restul modificari de matrice si dupa sa adaug la final
    # lungimea initiala a stringului si regulile de rotire (poate nu la final) dar dupa ce sunt gata facute ma folosesc iar de cifra de control si poate pun si ceva in fata
    # stringului criptat 

#string = "ana are mere"
#e_string = custom_encrypt_l2(string)
#print(e_string)



        

# def custom_encrypt_l2(text):
#     encryped_text = text
#     message_lenght = len(text)
#     matrix = string_to_matrix(encrypted_text)
#     rotate_rules = random_rule()
#     message_length_hidden = number_to_characters(message_length)

#     if message_len <= 15:
#         steps = random_in_interval(1, 4)
#         rotation_rules_hidden = random_rule()
#         rotation_rules = 

#         for layer in range(layers):
#         steps = steps_array[layer % len(steps_array)]  # Circular steps
#         direction_bits = direction_array[layer % len(direction_array)]  # Circular direction
#         direction = 'right' if direction_bits == 1 else 'left'
#         rotate_layer(matrix, layer, steps, direction)
        