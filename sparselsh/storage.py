from __future__ import print_function

import time
import pickle
import sys

try:
    import redis
except ImportError:
    redis = None

try:
    import bsddb
except ImportError:
    bsddb = None

try:
    import leveldb
except ImportError:
    leveldb = None

try:
    import cPickle as pickle
except ImportError:
    import pickle

def serialize( data):
    return pickle.dumps( data, protocol=2)
def deserialize( data):
    return pickle.loads( data)

__all__ = ['storage']

def storage(storage_config, index):
    """ Given the configuration for storage and the index, return the
    configured storage instance.
    """
    if 'dict' in storage_config:
        return InMemoryStorage(storage_config['dict'])
    elif 'redis' in storage_config:
        storage_config['redis']['db'] = index
        return RedisStorage(storage_config['redis'])
    elif 'berkeleydb' in storage_config:
        return BerkeleyDBStorage(storage_config['berkeleydb'])
    elif 'leveldb' in storage_config:
        return LevelDBStorage(storage_config['leveldb'])
    else:
        raise ValueError("Only in-memory dictionary, berkeleydb, leveldb, and redis are supported.")

class BaseStorage(object):
    def __init__(self, config):
        """ An abstract class used as an adapter for storages. """
        raise NotImplementedError

    def serialize( self, data):
        return serialize( data)

    def deserialize( self, data):
        return deserialize( data)

    def keys(self):
        """ Returns a list of binary hashes that are used as dict keys. """
        raise NotImplementedError

    def set_val(self, key, val):
        """ Set `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def get_val(self, key):
        """ Return `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def append_val(self, key, val):
        """ Append `val` to the list stored at `key`.

        If the key is not yet present in storage, create a list with `val` at
        `key`.
        """
        raise NotImplementedError

    def get_list(self, key):
        """ Returns a list stored in storage at `key`.

        This method should return a list of values stored at `key`. `[]` should
        be returned if the list is empty or if `key` is not present in storage.
        """
        raise NotImplementedError


class InMemoryStorage(BaseStorage):
    def __init__(self, config):
        self.name = 'dict'
        self.storage = dict()

    def keys(self):
        return list(self.storage.keys())

    def set_val(self, key, val):
        self.storage[key] = val

    def get_val(self, key):
        return self.storage[key]

    def append_val(self, key, val):
        self.storage.setdefault(key, []).append(val)

    def get_list(self, key):
        return self.storage.get(key, [])

class RedisStorage(BaseStorage):
    def __init__(self, config):
        if not redis:
            raise ImportError("redis-py is required to use Redis as storage.")
        self.name = 'redis'
        self.storage = redis.StrictRedis(**config)

    def keys(self, pattern="*"):
        return self.storage.keys(pattern)

    def set_val(self, key, val):
        self.storage.set(key, val)

    def get_val(self, key):
        return self.storage.get(key)

    def append_val(self, key, val):
        self.storage.rpush(key, serialize(val))

    def get_list(self, key):
        # TODO: find a better way to do this
        values = self.storage.lrange(key, 0, -1)
        return [ deserialize(v)for v in values]

class BerkeleyDBStorage(BaseStorage):
    def __init__(self, config):
        if 'filename' not in config:
            raise ValueError("You must supply a 'filename' in your config")
        self.storage = bsddb.hashopen( config['filename'])

    def __exit__(self, type, value, traceback):
        self.storage.sync()

    def keys(self):
        return list(self.storage.keys())

    def set_val(self, key, val):
        self.storage[key] = self.serialize(val)

    def get_val(self, key):
        return self.storage[key]

    def append_val(self, key, val):
        try:
            current = self.deserialize( self.storage[key])
        except KeyError:
            current = []

        # update new list
        current.append( val)
        self.storage[key] = self.serialize(current)

    def get_list(self, key):
        try:
            return self.deserialize(self.storage[key])
        except KeyError:
            return []

class LevelDBStorage(BaseStorage):
    def __init__(self, config):
        if not leveldb:
            raise ImportError("leveldb is required to use Redis as storage.")
        if 'db' not in config:
            raise ValueError("You must specify LevelDB filename as 'db' in your config")
        self.storage = leveldb.LevelDB( config['db'])

    def keys(self):
        return self.storage.RangeIter(include_value=False)

    def set_val(self, key, val):
        self.storage.Put( key, self.serialize(val))

    def get_val(self, key):
        return self.serialize( self.storage.Get( key))

    def append_val(self, key, val):
        # If a key doesn't exist, leveldb will throw KeyError
        current = []
        try:
            item = self.storage.Get(key)
            current = self.deserialize()
        except KeyError:
            pass
        except TypeError:
            # here, we have python3, which has a different manner for
            # interacting with leveldb
            if type(key) == str and sys.version_info[0] == 3:
                return self.append_val( bytes( key, 'UTF-8'), val)

        # update new list
        current.append( val)
        self.storage.Put(key, self.serialize(current))

    def get_list(self, key):
        try:
            k = self.storage.Get(key)
            return self.deserialize( k)
        except KeyError:
            return []
