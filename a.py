import sys
import struct
import mmap

def modify_coefficient(mm, n, value):
    marker = f"COEF_F{n}_".encode('ascii')
    pos = mm.find(marker)
    if pos == -1:
        print(f"Could not find marker for f{n}")
        return False
    mm.seek(pos + 8)
    mm.write(struct.pack('d', float(value)))
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python modify_coefs.py <coefficient_number> <value>")
        print("Example: python modify_coefs.py 1 2.5")
        sys.exit(1)
    
    n = int(sys.argv[1])
    value = float(sys.argv[2])
    
    binary_path = "./target/release/riw"  # adjust path to your binary
    
    with open(binary_path, 'rb+') as f:
        mm = mmap.mmap(f.fileno(), 0)
        if modify_coefficient(mm, n, value):
            print(f"Successfully modified f{n} to {value}")
        mm.flush()

if __name__ == '__main__':
    main()
