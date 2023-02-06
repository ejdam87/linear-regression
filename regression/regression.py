from vector import Vector

## ( input, expected_result )
Sample = tuple[ Vector, float ]

## --- Error function

## mean squared error
def mse( actual: list[ float ], expected: list[ float ] ) -> float:

    assert len( actual ) == len( expected )

    n = len( actual )
    res = 0
    for a, b in zip( actual, expected ):
        res += ( b - a ) ** 2

    return res / n

## partial derivation ( by k-th elementh of weight vector )
## of mse
def dmse( k: int,
          inputs: list[ Vector ],
          actual: list[ float ],
          expected: list[ float ] ) -> float:

    assert len( actual ) == len( expected ) == len( inputs )

    n = len( actual )
    res = 0
    for a, b, inpt in zip( actual, expected, inputs ):
        res += ( b - a ) * inpt[ k ]

    return ( -2 / n ) * res

## ---

## --- General model
class LinearRegression:
    def __init__( self, dim: int ) -> None:
        self.weights = Vector( [ 0 for _ in range( dim + 1 ) ] )    ## bias is weights[ 0 ]
        self.dim = dim
        self.learning_rate = 0.001

    ## Main method --> try to approximate expected value ( based on current weights )
    def approximate( self, v: Vector ) -> float:
        return self.weights * v.extended( 1 )

    def set_weights( self, weights: Vector ) -> None:
        self.weights = weights

    def get_weights( self ) -> Vector:
        return self.weights

    ## Error gradient calculation
    def gradient( self, samples: list[ Sample ] ) -> Vector:

        inputs = [ inpt.extended( 1 ) for inpt, expected in samples ]
        res = Vector( [ 0 for _ in range( self.dim + 1 ) ] )

        actual, expected = self._compute_approx( samples )
        for k in range( self.dim + 1 ):
            res[ k ] = dmse( k, inputs, actual, expected )

        return res

    def _compute_approx( self, samples: list[ Sample ] ) -> tuple[ list[ float ], list[ float ] ]:
        actual = []
        expected = []

        for v, exp in samples:
            actual.append( self.approximate( v ) )
            expected.append( exp )

        return actual, expected

    def regression_error( self, samples: list[ Sample ] ) -> float:
        actual, expected = self._compute_approx( samples )
        return mse( actual, expected )

    def learn( self, samples: list[ Sample ] ) -> None:

        EPOCHS = 10000
        for i in range( EPOCHS ):
            self.weights -= self.learning_rate * self.gradient( samples )

## ---


## --- Misc

import matplotlib.pyplot as plt
import numpy as np

import random

def create_random_dataset( count: int ) -> list[ Sample ]:

    upper = 10
    lower = 0

    res = []

    for x in range( count ):
        y = random.randint( lower, upper )
        res.append( ( vectorify( x ), y ) )

    return res

def vectorify( x: int ) -> Vector:
    return Vector( [ x ] )


def show_2d_regression( data: list[ Sample ],
                        model: LinearRegression ) -> None:
    
    xs = [ x[ 0 ] for x, _ in data ]
    ys = [ y for _, y in data ]

    plt.scatter( xs, ys, color="b" )

    xs = [ min( xs ), max( xs ) ]
    ys = [ model.approximate( vectorify( x ) ) for x in xs ]
    plt.plot( xs, ys, color="g" )
    plt.show( )


def show_3d_regression( data: list[ Sample ],
                        model: LinearRegression ) -> None:
    
    xs = [ x[ 0 ] for x, _ in data ]
    ys = [ x[ 1 ] for x, _ in data ]
    zs = [ z for _, z in data ]

    fig = plt.figure()
    ax = fig.add_subplot( projection="3d" )
    ax.scatter( xs, ys, zs )
    plt.show( )

dataset = [ ( Vector( [ 1, 3 ] ), 5 ) ]

model = LinearRegression( 2 )
model.set_weights( Vector( [ 0, 0 ] ) )

model.learn( dataset )

show_3d_regression( dataset, model )
