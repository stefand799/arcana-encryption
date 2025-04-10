#decrypt.py

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

#Unshuffle a text knowing the step
def unshuffle(text, step):
    length = len(text)
    first_part_length = (length + step - 1) // step
    first_part = text[:first_part_length]
    second_part = text[first_part_length:]

    result = []
    first_part_index = 0
    second_part_index = 0

    for i in range(length):
        if i % step == 0:
            result.append(first_part[first_part_index])
            first_part_index += 1
        else:
            result.append(second_part[second_part_index])
            second_part_index += 1

    return ''.join(result)


#Reverse cases up->low and low->up
def reverse_case(text):

      return ''.join([char.lower() if char.isupper() else char.upper() for char in text])

#Extracts chars from a string
def extract_char(text, index):
    if index < 0 or index >= len(text):
        raise ValueError("Index is out of bounds")

    extracted_char = text[index]
    
    remaining_text = text[:index] + text[index+1:]
    
    return extracted_char, remaining_text

#Unshift the chars in text based on dictionary search
def unshift(text, char_map):
    result = []
    char_map_len = len(char_map)
    
    for idx, char in enumerate(text):
        if char in char_map:
            index = char_map.index(char)
            shift_amount = idx  # Step is the index of char in text
            
            if idx % 2 == 0:
                shifted_index = (index - shift_amount) % char_map_len  # Shift to the right
            else:
                shifted_index = (index + shift_amount) % char_map_len  # Shift to the left
            
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

#Generate dictionary
def generate_dictionary(dictionary_rules, encrypt_factors):
    dictionary = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    factors = encrypt_factors

    for factor, d_rule in zip(encrypt_factors, dictionary_rules):

        dictionary=shuffle(dictionary, factor)

        if d_rule == '1':
            dictionary = reverse_case(dictionary)
        else:
            dictionary = cut_deck(dictionary)

    return dictionary


#Decrypt text
def custom_decrypt_l1(text):
    
    #Initialize values
    decrypted_text = text

    dictionary_char, decrypted_text = extract_char(decrypted_text, len(decrypted_text)-1)
    string_char, decrypted_text = extract_char(decrypted_text, 0)
    control_char, decrypted_text = extract_char(decrypted_text, digital_root(len(decrypted_text)-1))

    encrypt_factors=[2,3,5,7]
    dictionary_rules=bin(ord(dictionary_char))[2:].zfill(8)[:4]
    string_rules=bin(ord(string_char))[2:].zfill(8)[:4][::-1]
    control_rules=bin(ord(control_char))[2:].zfill(8)[:4][::-1]
    

    dictionary = generate_dictionary(dictionary_rules, encrypt_factors)
    encrypt_factors = encrypt_factors[::-1]
    dictionary_rules=dictionary_rules[::-1]

    #Decrypt
    for factor, d_rule, s_rule, c_rule in zip(encrypt_factors, dictionary_rules, string_rules, control_rules):
        
        decrypted_text = unshift(decrypted_text, dictionary)

        if c_rule == '1':
            decrypted_text = unshift(decrypted_text, dictionary)

        if s_rule == '1':
            decrypted_text = reverse_case(decrypted_text)
        else:
            decrypted_text = cut_deck(decrypted_text)
        
        if d_rule == '1':
            dictionary = reverse_case(dictionary)
        else:
            dictionary = cut_deck(dictionary)

        decrypted_text=unshuffle(decrypted_text, factor)
        dictionary=unshuffle(dictionary, factor)

    return decrypted_text


# # Functii matrice
# def fold_on_primary_diagonal(matrix):
#     n = len(matrix)
#     for i in range(n):
#         for j in range(i+1, n):
#             matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
#     return matrix

# def fold_on_secondary_diagonal(matrix):
#     n = len(matrix)
#     for i in range(n):
#         for j in range(n // 2):
#             opposite_j = n - 1 -j
#             matrix[i][j], matrix[i][opposite_j] = matrix[i][opposite_j], matrix[i][j]
#     return matrix

# def fold_horizontal(matrix):
#     n = len(matrix)
#     for i in range(n // 2):
#         matrix[i], matrix[n - 1 - i] = matrix[n - 1 - i], matrix[i]
#     return matrix

# def fold_vertical(matrix):
#     n = len(matrix)
#     m = len(matrix[0])
#     for i in range(n):
#         for j in range(m // 2):
#             matrix[i][j], matrix[i][m - 1 - j] = matrix[i][m - 1 - j]
#     return matrix

# #TREBUIE EDITATA
# def string_to_matrix(text):

#     mat_size = int(math.sqrt(len(text))) + 1
    
#     rows, cols = mat_size, mat_size
#     total_cells = row * cols

#     chars = list(text)

#     while len(chars) < total_cells:
#         random_char = random.choice(''.join(chr(i) for i in range(32, 127)))
#         chars.append(random_char)

#     matrix = []
#     for i in range(rows):
#         row = chars[i*cols:(i+1)*cols]
#         matrix.append(row)

#     return matrix

# def matrix_to_string(matrix):
#     long_string = ''.join(''.join(row) for row in matrix)
#     return long_string

# def rotate_layer(matrix, layer, steps, direction):
#     n = len(matrix)
#     if n == 0 or layer < 0 or layer >= (n + 1) // 2:
#         return matrix

#     # Calculate boundaries
#     top, left, bottom, right = layer, layer, n - layer - 1, n - layer - 1
#     layer_length = 2 * (right - left) + 2 * (bottom - top)

#     # Effective steps
#     steps = steps % layer_length
#     if direction == 'right':
#         steps = layer_length - steps  # Convert left rotation to equivalent right

#     # Perform the rotation in place
#     for _ in range(steps):
#         # Store the first element
#         temp = matrix[top][left]

#         # Move elements in the layer
#         for i in range(left, right):
#             matrix[top][i] = matrix[top][i + 1]
#         for i in range(top, bottom):
#             matrix[i][right] = matrix[i + 1][right]
#         for i in range(right, left, -1):
#             matrix[bottom][i] = matrix[bottom][i - 1]
#         for i in range(bottom, top, -1):
#             matrix[i][left] = matrix[i - 1][left]

#         # Place the first element in the last position
#         matrix[top + 1][left] = temp

#     return matrix

# def characters_to_number(chars):
#     if not (1 <= len(chars) <= 4):
#         raise ValueError("Input must contain 1 to 4 characters")

#     # Initialize the number
#     number = 0

#     # Assemble the number from the last 4 bits of each character
#     for i, char in enumerate(chars):
#         last_four_bits = ord(char) & 0x0F  # Get the last 4 bits
#         number |= last_four_bits << (i * 4)  # Shift and combine

#     return number

# def chars_to_number(chars):
#     if not 1 <= len(chars) <= 4:
#         raise ValueError("Input must contain between 1 and 4 characters")

#     # Convert each character into its binary representation (last 4 bits only)
#     bit_stacks = []
#     for char in chars:
#         # Get the last 4 bits of the character
#         char_bits = format(ord(char), '08b')[-4:]
#         bit_stacks.append(char_bits)

#     # Reverse the order of bit stacks to match the original stacking order
#     bit_stacks = bit_stacks[::-1]

#     # Combine the bits and convert back to a number
#     bit_string = ''.join(bit_stacks)
#     num = int(bit_string, 2)

#     return num