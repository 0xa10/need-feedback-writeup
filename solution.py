import numpy as np
import os
import pyshark
import numpy.matlib
import solve_lfsr
import sys
from collections import defaultdict
import math
CHALLENGE_PATH = "."
sys.path.append(CHALLENGE_PATH)
import channel
import glob

def flatten(l):
    return [x for sublist in l for x in sublist]

def bv(num, n):
    return bin(num)[2:].zfill(n)

def bv_list(num, n):
    return [int(i) for i in bv(num, n)][::-1]

def xor(data, key):
        return bytes([i ^ j for i,j in zip(data, key)])

def get_sbox_equations(sbox_tb):
    """Takes an sbox in the form of a list and returns a dictionary
    containing equation masks and results that have 100% bias for each output"""
    none = lambda x: all([i^1 for i in x]) # all not
    bitsum  = lambda x : bin(x)[2:].count('1') % 2 # sum of bits in number
    
    # Some parameters
    sbox_n_in = int(math.log(len(sbox_tb), 2))
    sbox_n_out = int(math.log(max(sbox_tb) + 1, 2))
    
    # Reverse the Sbox - each key in the dictionary is the list of inputs that lead to it
    sbox_reverse = defaultdict(list)
    for i, e in enumerate(sbox_tb):
        sbox_reverse[e].append(i)
    
    # For each Sbox output, attempt to find masks on the input which when summed always
    # lead to the same result
    masks = {out_val:defaultdict(int) for out_val in sbox_reverse.keys()} 
    for sbox_out,sbox_ins in sbox_reverse.items():
        for mask in range(1, 2**sbox_n_in):
            # Calculate sums
            sums = [bitsum(mask & sbox_in) for sbox_in in sbox_ins]
            if all(sums): # All sums are 1
                masks[sbox_out][mask] = 1
            elif none(sums): # All sums are 0
                masks[sbox_out][mask] = 0

    return masks

class KappaBreaker(object):
    def __init__(self, kappa):
        self._lfsr_coeffs = [
                bv_list(l.poly, l.n) for l in kappa.mlfsr.lfsrs
            ]
        self._lfsrs_N = (
                sum(len(l) for l in self._lfsr_coeffs) - len(self._lfsr_coeffs)
            )

        self._sbox_tb = kappa.mlfsr.sbox.tb
        self._sbox_n_in = int(math.log(len(self._sbox_tb), 2))
        self._sbox_n_out = int(math.log(max(self._sbox_tb) + 1, 2))
        self._masks = get_sbox_equations(self._sbox_tb)

        self._cipher_bytes = bytes()
        self._known_idx = 0
        self.equations = []
        
    
    def add_cipher_bytes(self, cipher_bytes):
        self._cipher_bytes += cipher_bytes
    
    def feed_known(self, known, idx=None):
        if idx:
            self._known_idx = idx
        self._process_keystream(
                xor(known, self._cipher_bytes[self._known_idx:]),
                self._known_idx
            )
        self._known_idx += len(known)
    
    def _process_keystream(self, keystream, start_index):
        keystream_nibbles = flatten((i&0xF, i>>4) for i in keystream)
        start_index_nibble = start_index * 2
        
        for nibble_index, nibble in enumerate(keystream_nibbles):
            self._process_keystream_nibble(nibble, start_index_nibble + nibble_index)
    
    def _process_keystream_nibble(self, nibble, index):
        # The index tells us which 6 equations we need from the equation set
        all_lfsr_equations = [
                solve_lfsr.get_lfsr_equations(l, 6*(index+1))[:,-6:] 
                for l in self._lfsr_coeffs
            ]
        frame = np.vstack(all_lfsr_equations).astype(int)
        assert frame.shape[0] == self._lfsrs_N
        
        for mask, result in self._masks[nibble].items():
            mask_matrix = np.vstack([[bv_list(mask, self._sbox_n_in)],] * self._lfsrs_N)
            assert mask_matrix.shape == frame.shape
            
            summed_equation = np.bitwise_xor.reduce(
                    np.multiply(mask_matrix, frame), 
                    axis=1
                )
            self.equations.append(
                    np.concatenate((summed_equation, np.matrix(result)), axis=1).tolist()[0]
                )
    
    def can_decrypt(self):
        return (get_matrix_rank_gf2(self.equations) > self._lfsrs_N)

    def get_decryptor(self):
        if not self.can_decrypt():
            raise Exception("Not enough equations!")
        
        eqs = gaussian_elimination_gf2(self.equations)
         
        # generate dummy KappaCrypto, replace lfsr state, and return it
        k = channel.KappaCrypto("")

        ssvec = [_[0] for _ in eqs[:self._lfsrs_N+1,-1].tolist()]
        new_state = ssvec[:10], ssvec[10:21], ssvec[21:33], ssvec[33:46], ssvec[46:]        
        
        for a,b in zip(k.mlfsr.lfsrs, new_state):        
            a.state = solve_lfsr.bitarray_to_dec(b)
        
        return k

def get_matrix_rank_gf2(equations):
    reduced = gaussian_elimination_gf2(equations)
    
    # Count how many columns we have where the coefficients have been isolated (just 1)
    rank = 1
    for row in reduced:
        if np.sum(row) != 0:
            rank += 1
    return rank

def gaussian_elimination_gf2(equations):
    reduced = np.matrix(equations)
    
    for column in range(reduced.shape[1]):
        pivot_row = None
        target_row = None
        # Find pivot row (first row with non-zero element in current column)
        for row in range(column, reduced.shape[0], 1):
            if reduced[row, column] == 1:
                pivot_row = row 
                target_row = column
                break
        else:
            # No pivot row for this column
            assert np.sum(reduced[column:,column]) == 0
            continue
        assert pivot_row is not None
        assert target_row is not None

        # Found a pivot row for column 
        # Swap pivot row and current row
        reduced[[target_row, pivot_row],column:] = reduced[[pivot_row, target_row],column:]
        
        # Now go over all rows except the current row and xor the ones with 1's in 
        # the current column
        for row in range(reduced.shape[0]):
            if row == target_row:
                continue
            if reduced[row, column] == 1:
                reduced[row,column:] ^= reduced[target_row,column:]
     
    return reduced 
   
def follow_tcp_stream(pcap, stream_index):
    stream = [p for p in pcap if p.tcp.stream == stream_index]
    stream = sorted(stream, key = lambda x: int(x.number))
    assert bool(int(stream[0].tcp.flags_syn))
    assert bool(int(stream[1].tcp.flags_syn)) and bool(int(stream[1].tcp.flags_ack))
    assert bool(int(stream[2].tcp.flags_ack))
    
    data = defaultdict(bytes)
    for packet in stream[3:]:
        direction = "%s.%s-%s.%s" % (
                packet.ip.src, packet.tcp.srcport,
                packet.ip.dst, packet.tcp.dstport
            )
        if bool(int(packet.tcp.len)) and (len(data[direction]) < int(packet.tcp.seq)): # If packet is carrying data and is not a retrans
            payload = bytes.fromhex(packet.tcp.payload.raw_value)
            data[direction] += payload
            assert (int(packet.tcp.seq) + int(packet.tcp.len) - 1) == len(data[direction]) # no jitters plz
    return data

def main():
    # Taken from data.pcap.cleaned, first response from server
    real_buf = b'\xcf\x00\x00\x00\x02\x00\x00\x00\x8dr\xde\xa3F\xf4g\xf1ip\x9e\xb8~Sp\xa5&\x81-\xc3JzM\x96\xb5h\x9dTaA\x95;\x11\xdf\xf4\xd6\x08(;\xb2\xca\xe9\xf1\x8bZx\x11\x1fs|\xff{,\x86#1\x13{\x8f\xe4\xa3ACq\x0ck\x02\x9e\xcc\xae":8H\xdc:\x0f\x05\xa1\xbf\xe3\x8f\xd4\xbd=\x02\xc1\x93\\\x0e?w\xe1\xb6\xb3I\xfb\xc2=\x9a\x0e\x1b\x96\xae <\xf4G\xa5\x8f\x81\x87\xfe"\xc7\xd0\x0emp\xec#\r\x9b\x1d|\xefO\xab\xa1\ta)\x8e+{\x86\xe9\x08\xf4L\x8b\x14\xe759\x90-V\xe51\x94\xa1\x1f%\xaez\xdd\x83\x9e\xa8\xa4\x0b\xd9X\xb0[\n\x0cw\xba"\x1c\x19\xc8V\t\x8b\x9bG0\x03\xee\xdc\x15\xd1\xcf-Nh\xb4\x8b\xc9\xe3\xc6\xf9\xaavI\xa4S\x1d\x1d\xa8'
    # Use Kappa objects to remove header
    kp = channel.KappaPacket()
    kp.append(real_buf)
    kb = kp.extract_one()
    
    # Generate a KappaCrypto object (with the same Sbox and coefficients
    kappa = channel.KappaCrypto("") # seed doesnt matter
    breaker = KappaBreaker(kappa)
    
    # Use cipher bytes from packets and guessed known to generate keystream
    breaker.add_cipher_bytes(kb.data)
    known = b"HTTP/1.0 200 OK\r\n"    

    breaker.feed_known(known)
    print("Got %d equations, attempting to solve for original state" % (len(breaker.equations),))
    if not breaker.can_decrypt():
        print("Not enough keystream!")
        return -1
     
    decryptor = breaker.get_decryptor()
    for i, lfsr in enumerate(decryptor.mlfsr.lfsrs):
        print("LFSR %d state:\n\t0x%x" % (i+1, lfsr.state))

    plain = decryptor.proc(kb.data, is_enc=True)
    if not plain.startswith(known):
        print("Failed to recover LFSR state, check equations")
        return -1
    
    # success!
    print("Test decryption successful!")

    decryptor = breaker.get_decryptor() # reset decryptor

    # Continue decrypting info from packet
    print("Loading packets...")
    pcap = pyshark.FileCapture(os.path.join(CHALLENGE_PATH, "tmp/data.pcap.cleaned"))

    # Get all tcp streams, sorted by number
    tcp_streams = sorted(
            set(p.tcp.stream for p in pcap if p.transport_layer == "TCP"),
            key=int
        )
    print("Filtering streams (this may take a while)...")
    streams = [follow_tcp_stream(pcap, i) for i in tcp_streams]
    assert all(len(flatten(i.values())) for i in streams)
    for stream in streams:
        # Should only be two directions - to the server (target port 2346)
        directions = tuple(stream.keys())
        assert len(directions) == 2
        to_server, from_server = (
                directions if directions[0].endswith(str(channel.SERVER_TUNNEL_PORT)) 
                else directions[::-1]
            )
        
        # Extract unencrypted request
        kp = channel.KappaPacket()
        # to_server communications are unencrypted, retrieve request
        kp.append(stream[to_server])
        request = kp.extract_one()
        request_data = request.data.splitlines()[0].split()
        assert request_data[0] == b'GET'

        filename = os.path.basename(request_data[1].decode("utf-8"))
        print("Found request for file %r, extracting encrypted info from response" % (filename,)) 
         
        
        # Decrypt into file by looping over all to_server data
        print("Decrypting into file")
        kp = channel.KappaPacket()
        kp.append(stream[from_server])
        encrypted = kp.extract_one()
        with open(os.path.join(CHALLENGE_PATH, "tmp/%s" % filename), "wb") as f:
            while encrypted:
                assert encrypted.typ == channel.MsgType.data
                f.write(decryptor.proc(encrypted.data, is_enc=True))
                encrypted = kp.extract_one()
        
    print("Done!")
    return 0 
     
if __name__ == "__main__":
     sys.exit(main())
