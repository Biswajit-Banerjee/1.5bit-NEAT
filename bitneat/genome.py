import random
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from neat import DefaultGenome
import struct

# classes for bit quantized
class BitNodeGene(DefaultNodeGene):
    def __init__(self, key):
        super().__init__(key)
        self.bias = 0.0  # No bias

class BitConnectionGene(DefaultConnectionGene):
    def mutate(self, config):
        self.weight = BitGenome.quantize_weight(self.weight)


class BitGenome(DefaultGenome):
    @staticmethod
    def quantize_weight(weight):
        return round(weight, 0)

    @staticmethod
    def create_node(config, node_id):
        node = BitNodeGene(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = BitConnectionGene((input_id, output_id))
        connection.init_attributes(config)
        connection.weight = BitGenome.quantize_weight(random.uniform(-1, 1))
        return connection

    def mutate(self, config):
        super().mutate(config)
        for cg in self.connections.values():
            cg.weight = self.quantize_weight(cg.weight)
            
    def store(self):
        """
        Store the genome in a compact binary format.
        Returns a bytes object representing the genome.
        """
        num_connections = len(self.connections)
        weight_bits = ''.join(
            '{:02b}'.format(self.quantize_weight(cg.weight) + 1) for cg in self.connections.values()
        )
        num_weight_bytes = (num_connections + 7) // 8  # Ceiling division by 8
        weight_bytes = int(weight_bits, 2).to_bytes(num_weight_bytes, byteorder='big')
        return struct.pack('I', num_connections) + weight_bytes

    @classmethod
    def load(cls, data):
        """
        Load a BitGenome from a compact binary format.
        Args:
            data (bytes): The binary data representing the genome.
        Returns:
            A BitGenome instance.
        """
        num_connections = struct.unpack('I', data[:4])[0]
        data = data[4:]
        weight_bits = ''.join('{:08b}'.format(byte) for byte in data)[:num_connections * 2]
        weights = [int(weight_bits[i:i + 2], 2) - 1 for i in range(0, len(weight_bits), 2)]
        genome = cls()
        for i, weight in enumerate(weights):
            genome.connections[i] = BitConnectionGene((i, i + 1))
            genome.connections[i].weight = weight
        return genome
            
# classes for int quantized 
class IntNodeGene(DefaultNodeGene):
    def __init__(self, key):
        super().__init__(key)
        self.bias = IntGenome.quantize_weight(random.uniform(-1e4, 1e4))

class IntConnectionGene(DefaultConnectionGene):
    def mutate(self, config):
        self.weight = IntGenome.quantize_weight(self.weight)


class IntGenome(DefaultGenome):
    @staticmethod
    def quantize_weight(weight):
        return round(weight, 0)

    @staticmethod
    def create_node(config, node_id):
        node = IntNodeGene(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = IntConnectionGene((input_id, output_id))
        connection.init_attributes(config)
        connection.weight = IntGenome.quantize_weight(random.uniform(-1e4, 1e4))
        return connection

    def mutate(self, config):
        super().mutate(config)
        for cg in self.connections.values():
            cg.weight = self.quantize_weight(cg.weight)
            
    def store(self):
        """
        Store the genome in a compact binary format.
        Returns a bytes object representing the genome.
        """
        num_connections = len(self.connections)
        weight_bytes = b''.join(
            struct.pack('i', self.quantize_weight(cg.weight)) for cg in self.connections.values()
        )
        return struct.pack('I', num_connections) + weight_bytes

    @classmethod
    def load(cls, data):
        """
        Load an IntGenome from a compact binary format.
        Args:
            data (bytes): The binary data representing the genome.
        Returns:
            An IntGenome instance.
        """
        num_connections = struct.unpack('I', data[:4])[0]
        data = data[4:]
        weights = [
            struct.unpack('i', data[i:i + 4])[0] for i in range(0, len(data), 4)
        ]
        genome = cls()
        for i, weight in enumerate(weights):
            genome.connections[i] = IntConnectionGene((i, i + 1))
            genome.connections[i].weight = weight
        return genome
