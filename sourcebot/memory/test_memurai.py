"""
Test Memurai Connection
-----------------------
Run this to verify Memurai is working with Python.
"""

import redis


def test_connection () :
    """Test basic Redis/Memurai connection"""
    print( "Testing Memurai connection..." )
    print( "-" * 50 )

    try :
        # Connect to Memurai (uses same port as Redis: 6379)
        r = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True,
            socket_connect_timeout=5
        )

        # Test 1: Ping
        print( "Test 1: Ping" )
        response = r.ping()
        print( f"  ✅ Ping successful: {response}" )

        # Test 2: Set a value
        print( "\nTest 2: Set value" )
        r.set( 'test_key', 'Hello from SourceUp!' )
        print( "  ✅ Value set successfully" )

        # Test 3: Get the value
        print( "\nTest 3: Get value" )
        value = r.get( 'test_key' )
        print( f"  ✅ Retrieved value: {value}" )

        # Test 4: JSON storage (what we'll use for sessions)
        print( "\nTest 4: JSON storage" )
        import json
        test_session = {
            'user_id' : 'user123',
            'product' : 'LED bulbs',
            'location' : 'Vietnam',
            'max_price' : 2.0
        }
        r.set( 'session_test', json.dumps( test_session ) )
        retrieved = json.loads( r.get( 'session_test' ) )
        print( f"  ✅ JSON storage works: {retrieved}" )

        # Test 5: Delete keys
        print( "\nTest 5: Cleanup" )
        r.delete( 'test_key', 'session_test' )
        print( "  ✅ Cleanup successful" )

        print( "\n" + "=" * 50 )
        print( "🎉 ALL TESTS PASSED!" )
        print( "Memurai is ready to use with SourceUp!" )
        print( "=" * 50 )

        return True

    except redis.ConnectionError as e :
        print( f"\n❌ Connection Error: {e}" )
        print( "\nTroubleshooting:" )
        print( "1. Check if Memurai service is running:" )
        print( "   sc query Memurai" )
        print( "2. If not running, start it:" )
        print( "   sc start Memurai" )
        print( "3. Or restart the Memurai service from Services app" )
        return False

    except Exception as e :
        print( f"\n❌ Unexpected Error: {e}" )
        return False


if __name__ == "__main__" :
    test_connection()