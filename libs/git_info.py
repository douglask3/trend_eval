## return basic git info for adding to plots/ouputs ##
from os import popen

rev = popen('git rev-parse HEAD').read()[0:7]
url = popen('git config --get remote.origin.url').read()


#    def rev(self):
#        return popen('git rev-parse head').read()[0:7]
#
#    def url(self):
#        return popen('git config --get remote.origin.url').read()
#
#    def git(self):
#        return 'repo: ' + url() + '\n' + 'rev:  ' + rev()


