import json, requests
import logging
import psutil

class StanfordCoreNLP:

    def __init__(self, server_url):
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url

    def annotate(self, text, properties=None):
        assert isinstance(text, str)
        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)

        # Checks that the Stanford CoreNLP server is started.
        try:
            requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the CoreNLP server e.g.\n'
            '$ cd stanford-corenlp-full-2015-12-09/ \n'
            '$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer')

        data = text.encode()
        r = requests.post(
            self.server_url, params={
                'properties': str(properties)
            }, data=data, headers={'Connection': 'close'})
        output = r.text
        if ('outputFormat' in properties
             and properties['outputFormat'] == 'json'):
            try:
                output = json.loads(output, encoding='utf-8', strict=True)
            except:
                pass
        return output

    def tokensregex(self, text, pattern, filter):
        return self.regex('/tokensregex', text, pattern, filter)

    def semgrex(self, text, pattern, filter):
        return self.regex('/semgrex', text, pattern, filter)

    def regex(self, endpoint, text, pattern, filter):
        r = requests.get(
            self.server_url + endpoint, params={
                'pattern':  pattern,
                'filter': filter
            }, data=text)
        output = r.text
        try:
            output = json.loads(r.text)
        except:
            pass
        return output

    def close(self):
        logging.info('Cleanup...')
        if hasattr(self, 'p'):
            try:
                parent = psutil.Process(self.p.pid)
            except psutil.NoSuchProcess:
                logging.info('No process: {}'.format(self.p.pid))
                return

            if self.class_path_dir not in ' '.join(parent.cmdline()):
                logging.info('Process not in: {}'.format(parent.cmdline()))
                return

            children = parent.children(recursive=True)
            for process in children:
                logging.info('Killing pid: {}, cmdline: {}'.format(process.pid, process.cmdline()))
                # process.send_signal(signal.SIGTERM)
                process.kill()

            logging.info('Killing shell pid: {}, cmdline: {}'.format(parent.pid, parent.cmdline()))
            # parent.send_signal(signal.SIGTERM)
            parent.kill()
