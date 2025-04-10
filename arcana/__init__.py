import os
import argparse
import sys
import encrypt  # Assuming encrypt.custom_encrypt_l1 exists
import decrypt  # Assuming decrypt.custom_decrypt_l1 exists

def process_file(input_file, output_file, chunk_size=20):
    """
    Encrypts the input file line by line. Each line is split into chunks,
    encrypted, and written to the output file on the same line.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()  # Remove leading/trailing spaces or newline
            # Break the line into chunks of `chunk_size`
            chunks = [line[i:i + chunk_size] for i in range(0, len(line), chunk_size)]
            # Encrypt each chunk and collect the results
            encrypted_chunks = [encrypt.custom_encrypt_l1(chunk) for chunk in chunks]
            # Write the encrypted chunks to the file, joined by spaces, on the same line
            outfile.write("".join(encrypted_chunks) + "\n")

def decrypt_file(input_file, output_file, chunk_size=20):
    """
    Decrypts the input file line by line. Each line is split into adjusted-size chunks,
    decrypted, and written to the output file on the same line.
    """
    adjusted_chunk_size = chunk_size + 3  # Account for the 3-character overhead during encryption
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()  # Remove leading/trailing spaces or newline
            # Split the line into chunks of `adjusted_chunk_size`
            encrypted_chunks = [line[i:i + adjusted_chunk_size] for i in range(0, len(line), adjusted_chunk_size)]
            # Decrypt each chunk and collect the results
            decrypted_chunks = [decrypt.custom_decrypt_l1(chunk) for chunk in encrypted_chunks]
            # Write the decrypted chunks to the file, concatenated as one line
            outfile.write("".join(decrypted_chunks) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Encrypt or decrypt a text file.")
    parser.add_argument(
        "mode", choices=["encrypt", "decrypt"], help="Choose whether to encrypt or decrypt the file."
    )
    parser.add_argument("input_file", help="Path to the input file.")
    parser.add_argument("output_file", help="Path to the output file.")
    parser.add_argument(
        "--chunk-size", type=int, default=20, help="Chunk size for processing (default: 20 characters)."
    )

    args = parser.parse_args()

    # Check if the input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        if args.mode == "encrypt":
            process_file(args.input_file, args.output_file, args.chunk_size)
            print(f"File successfully encrypted and saved to '{args.output_file}'.")
        elif args.mode == "decrypt":
            decrypt_file(args.input_file, args.output_file, args.chunk_size)
            print(f"File successfully decrypted and saved to '{args.output_file}'.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()