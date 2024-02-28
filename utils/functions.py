def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys(): dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str
class Storage(dict):
    def __getattr__(self, key):
        try:
            return self[key] if key in self else False
        except KeyError as k:
            raise AttributeError(k)
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)
    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"