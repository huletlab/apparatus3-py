#!/usr/bin/python

# ----------------------------------------
# image_viewer2.py
#
# Created 03-20-2010
#
# Author: Mike Driscoll
# ----------------------------------------

import glob
import os
import wx
from wx.lib.pubsub import Publisher

import sys
sys.path.append('/lab/software/apparatus3/bin/py')
import falsecolor, fits2png, manta2png

########################################################################
class ViewerPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        
        #width, height = wx.DisplaySize()
        width, height = (700,850)
        self.picPaths = []
        self.currentPicture = 0
        self.totalPictures = 0
        self.photoMaxSize = height - 200
        Publisher().subscribe(self.updateImages, ("update images"))

        self.slideTimer = wx.Timer(None)
        self.slideTimer.Bind(wx.EVT_TIMER, self.update)
        
        self.layout()
        
    #----------------------------------------------------------------------
    def layout(self):
        """
        Layout the widgets on the panel
        """
        
        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        
        img = wx.EmptyImage(self.photoMaxSize,self.photoMaxSize)
        self.imageCtrl = wx.StaticBitmap(self, wx.ID_ANY, 
                                         wx.BitmapFromImage(img))
        self.mainSizer.Add(self.imageCtrl, 0, wx.ALL|wx.CENTER, 5)
        self.imageLabel = wx.StaticText(self, label="")
        self.mainSizer.Add(self.imageLabel, 0, wx.ALL|wx.CENTER, 5)
        
        btnData = [("Previous", btnSizer, self.onPrevious),
                   ("Slide Show", btnSizer, self.onSlideShow),
                   ("Next", btnSizer, self.onNext)]
        for data in btnData:
            label, sizer, handler = data
            self.btnBuilder(label, sizer, handler)
            
        self.mainSizer.Add(btnSizer, 0, wx.CENTER)
        self.SetSizer(self.mainSizer)
            
    #----------------------------------------------------------------------
    def btnBuilder(self, label, sizer, handler):
        """
        Builds a button, binds it to an event handler and adds it to a sizer
        """
        btn = wx.Button(self, label=label)
        btn.Bind(wx.EVT_BUTTON, handler)
        sizer.Add(btn, 0, wx.ALL|wx.CENTER, 5)
        
    #----------------------------------------------------------------------
    def loadImage(self, image):
        """"""
        atomsfile = image
        shot = atomsfile.split('atoms')[0]
        type = atomsfile.split('atoms')[1]

        pngpath = shot + '_falsecolor.png'
        if not os.path.exists( pngpath ):
            print 'Creating ' + pngpath + '...'
            if type == '.fits':
                pngpath = fits2png.makepng( atomsfile, 'ABS', 80)
            if type == '.manta': 
                pngpath = manta2png.makepng( atomsfile, 'PHC', 80)

        image_name = os.path.basename(pngpath)
        img = wx.Image(pngpath, wx.BITMAP_TYPE_ANY)
        # scale the image, preserving the aspect ratio
        W = img.GetWidth()
        H = img.GetHeight()
        if W > H:
            NewW = self.photoMaxSize
            NewH = self.photoMaxSize * H / W
        else:
            NewH = self.photoMaxSize
            NewW = self.photoMaxSize * W / H
        img = img.Scale(NewW,NewH)

        self.imageCtrl.SetBitmap(wx.BitmapFromImage(img))
        self.imageLabel.SetLabel(image_name)
        self.Refresh()
        Publisher().sendMessage("resize", "")
        
    #----------------------------------------------------------------------
    def nextPicture(self):
        """
        Loads the next picture in the directory
        """
        if self.currentPicture == self.totalPictures-1:
            self.currentPicture = 0
        else:
            self.currentPicture += 1
        self.loadImage(self.picPaths[self.currentPicture])
        
    #----------------------------------------------------------------------
    def previousPicture(self):
        """
        Displays the previous picture in the directory
        """
        if self.currentPicture == 0:
            self.currentPicture = self.totalPictures - 1
        else:
            self.currentPicture -= 1
        self.loadImage(self.picPaths[self.currentPicture])
        
    #----------------------------------------------------------------------
    def update(self, event):
        """
        Called when the slideTimer's timer event fires. Loads the next
        picture from the folder by calling th nextPicture method
        """
        self.nextPicture()
        
    #----------------------------------------------------------------------
    def updateImages(self, msg):
        """
        Updates the picPaths list to contain the current folder's images
        """
        self.picPaths = msg.data
        self.totalPictures = len(self.picPaths)
        self.loadImage(self.picPaths[0])
        
    #----------------------------------------------------------------------
    def onNext(self, event):
        """
        Calls the nextPicture method
        """
        self.nextPicture()
    
    #----------------------------------------------------------------------
    def onPrevious(self, event):
        """
        Calls the previousPicture method
        """
        self.previousPicture()
    
    #----------------------------------------------------------------------
    def onSlideShow(self, event):
        """
        Starts and stops the slideshow
        """
        btn = event.GetEventObject()
        label = btn.GetLabel()
        if label == "Slide Show":
            self.slideTimer.Start(1000)
            btn.SetLabel("Stop")
        else:
            self.slideTimer.Stop()
            btn.SetLabel("Slide Show")
        
        
########################################################################
class ViewerFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Image Viewer")
        panel = ViewerPanel(self)
        self.folderPath = ""
        Publisher().subscribe(self.resizeFrame, ("resize"))
        
        self.initToolbar()
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(self.sizer)
        
        self.Show()
        self.sizer.Fit(self)
        self.Center()
        self.onOpenDirectory(0)
        
        
    #----------------------------------------------------------------------
    def initToolbar(self):
        """
        Initialize the toolbar
        """
        self.toolbar = self.CreateToolBar()
        self.toolbar.SetToolBitmapSize((16,16))
        
        open_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16,16))
        openTool = self.toolbar.AddSimpleTool(wx.ID_ANY, open_ico, "Open", "Open an Image Directory")
        self.Bind(wx.EVT_MENU, self.onOpenDirectory, openTool)
        
        self.toolbar.Realize()
        
    #----------------------------------------------------------------------
    def onOpenDirectory(self, event):
        """
        Clicking this button refreshes the list of picPaths
        """
        print os.getcwd()
        #dlg = wx.DirDialog(self, "Choose a directory",
        #                   style=wx.DD_DEFAULT_STYLE)
        
        #if dlg.ShowModal() == wx.ID_OK:
        #    self.folderPath = dlg.GetPath()
        #    print self.folderPath
        #    picPaths = glob.glob(self.folderPath + "/????atoms.fits")
  
        print "type = " + image_type
        picPaths = glob.glob( os.getcwd() + "/????atoms" + image_type)
        picPaths.sort()
        print picPaths
        if picPaths == []:
            print " ---> There are no images of the specified type"
            print " ---> Program will exit" 
            exit(1)

        Publisher().sendMessage("update images", picPaths)
        
    #----------------------------------------------------------------------
    def resizeFrame(self, msg):
        """"""
        self.sizer.Fit(self)
        
#----------------------------------------------------------------------
if __name__ == "__main__":
    #parameter indices
    IMAGETYPE = 1
    
    if not (len(sys.argv) == IMAGETYPE + 1):
        print "viewer.py"
        print ""     
        print "Starts the image viewer for images of the specified extension."     
        print "Extension can be .fits or .manta"     
        print ""     
        print "usage: viewer.py .fits"     
        print "usage: viewer.py .manta"
        exit(1)

    image_type = sys.argv[IMAGETYPE]
    print image_type 
    if image_type != ".fits" and image_type != ".manta":
        print " ---> The image type used is not valid!"
        exit(1)
     
 
    app = wx.PySimpleApp()
    frame = ViewerFrame()
    app.MainLoop()
    
