import logging

def init_log(dir = 'logs'):
    logging.basicConfig(filename = '/'.join(__file__.split('/')[:-1]) + f'/{dir}/log.txt', 
                    filemode = 'a',
                    datefmt = "%H:%M:%S", 
                    level = logging.INFO,
                    format = '%(asctime)s[%(levelname)s] ~ %(message)s')
    write_log('\n\n' + "VV" * 40 + '\n' + " " * 37 + 'NEW LOG\n' + "^^" * 40)
    
def write_log(info = None, c_tag = None, verbose = True):
    if verbose:
        print(str(info))
        
    if c_tag is not None:
        logging.info(str(c_tag) + ' ' + str(info))
    else:
         logging.info(str(info))
    


import json
class data_recorder:
    def __init__(self, filename):
        self.data_dict = {}
        self._set_record_filename(filename)

    def _set_record_filename(self, filename='data_record.json'):
        self.record_name = filename
        self.path =  '/'.join(__file__.split('/')[:-1]) + f'/data_records/{filename}'

    def add_record(self, name, data):
        if name not in self.data_dict:
            self.data_dict[name] = []
        self.data_dict[name].append(data)

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data_dict, f)

def get_data_from_record(filename):
    path =  '/'.join(__file__.split('/')[:-1]) + f'/data_records/{filename}'
    with open(path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    dr = data_recorder()
    dr.set_record_filename('sampling_noise.json')
    dr.add_record('a', 1)
    dr.add_record('a', 2)
    dr.add_record('b', 3)
    dr.add_record('b', 4)
    dr.add_record('a', 1)
    dr.add_record('a', 2)
    dr.add_record('b', 3)
    dr.add_record('b', 4)
    dr.add_record('a', 1)
    dr.add_record('a', 2)
    dr.add_record('b', 3)
    dr.add_record('b', 4)
    dr.add_record('a', 1)
    dr.add_record('a', 2)
    dr.add_record('b', 3)
    dr.add_record('b', 4)
    dr.save()
    data = get_data_from_record('sampling_noise.json')
    print(data)
