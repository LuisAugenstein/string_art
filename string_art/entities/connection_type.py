
N_CONNECTION_TYPES = 4


class ConnectionType:
    STRAIGHT_IN = 0
    STRAIGHT_OUT = 1
    DIAGONAL_IN = 2
    DIAGONAL_OUT = 3


connection_type_masks = {
    'ingoing': lambda connection_types: (connection_types == ConnectionType.STRAIGHT_IN) | (connection_types == ConnectionType.DIAGONAL_IN),
    'outgoing': lambda connection_types: (connection_types == ConnectionType.STRAIGHT_OUT) | (connection_types == ConnectionType.DIAGONAL_OUT),
    'straight': lambda connection_types: (connection_types == ConnectionType.STRAIGHT_IN) | (connection_types == ConnectionType.STRAIGHT_OUT),
    'diagonal': lambda connection_types: (connection_types == ConnectionType.DIAGONAL_IN) | (connection_types == ConnectionType.DIAGONAL_OUT),
}
