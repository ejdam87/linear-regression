from vector import Vector
from random import randint

## ( input, expected_result )
Sample = tuple[ Vector, float ]

def create_x( dim: int, lower: int, upper: int ) -> Vector:
    return Vector( [ randint( lower, upper ) for _ in range( dim ) ] )


def create_dataset( size: int,
                    dim: int,
                    lower_x: int=-10,
                    upper_x: int=10,
                    lower_y: int=-10,
                    upper_y: int=10 ) -> list[ Sample ]:
    
    res = []
    for _ in range( size ):
        y = randint( lower_y, upper_y )
        x = create_x( dim, lower_x, upper_x )
        res.append( ( x, y ) )

    return res
