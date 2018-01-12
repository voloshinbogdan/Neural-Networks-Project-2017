# Notifier example from tutorial
#
# See: http://github.com/seb-m/pyinotify/wiki/Tutorial
#
import pyinotify

wm = pyinotify.WatchManager()  # Watch Manager
mask = pyinotify.IN_DELETE | pyinotify.IN_CREATE  # watched events

class EventHandler(pyinotify.ProcessEvent):
    def process_IN_CREATE(self, event):
        print "Creating:", event.name

    def process_IN_DELETE(self, event):
        print "Removing:", event.name

handler = EventHandler()
notifier = pyinotify.Notifier(wm, handler)
wdd = wm.add_watch('/home/wisestump/', mask, rec=True)

notifier.loop()
