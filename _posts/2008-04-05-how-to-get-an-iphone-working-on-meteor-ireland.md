---
id: 21
title: How to get an iPhone working on Meteor Ireland
date: 2008-04-05T21:12:32+00:00
author: Ken
layout: single

permalink: /how-to-get-an-iphone-working-on-meteor-ireland/
categories:
  - Computers
  - Macintosh
---
This is my experience with getting an iPhone working with Ireland&#8217;s Meteor network. Currently the iPhone is only available from O2, and only with an 18-month service contract. I have been using a simple, pay-as-you-go plan from [Meteor](http://www.meteor.ie/personal/pay_as_you_go/ "Meteor Ireland's Pay As You Go") and wanted to use that in an iPhone. Here are the steps I took to get it working.

  1. **Buy an iPhone.** Not so easy to do in Ireland, but you can find them on eBay, buy them from an Apple Store (in the US or say, London), or talk your way into getting an AT&T store to selling you one without requiring you to sign the activation contract in the store. (Tell them it&#8217;s for a friend who will activate it Apple-Store-style using iTunes from home.) 
  2. **Jailbreak and unlock the iPhone.** This used to be complicated but now it is ultra-easy thanks to the excellent utility [ZiPhone](http://www.ziphone.org/ "ZiPhone"). My phone came with version 1.1.4 of the iPhone software, and I used ZiPhone 3.0 (30th of March 2008). You simply connect your iPhone to your computer with the included USB cable, turn off iTunes once it starts up, and run ZiPhone. (I was using the Mac version, but the Windows version is supposed to work similarly.) I chose the first option, &#8220;Unlock, Jailbreak, and Activate&#8221;. You get a message telling you to wait 4 mins 30 secs, a bunch of stuff scrolls across your iPhone, it restarts a couple of times, and at the end of the process, you have an activated and unlocked iPhone. ZiPhone also automatically installs a piece of software called Installer.app (3.01 on mine, from Nullriver Software), plus the BSD Subsystem and OpenSSH. 
  3. **Remove the AT&T SIM card included with the iPhone and replace it with your own SIM card.** There is a pin hole you need to push on the top of the phone with a paper clip, and the SIM tray slides out. You simply replace the existing card with your own, and then reinsert. There are plenty of videos showing how to do this on YouTube, for example [this one](http://www.youtube.com/watch?v=jaTb_TJTiAM). You can then toss out the AT&T card or keep it for a souvenir. Your iPhone should be working fine with Meteor now. It may ask you to enter the PIN for your SIM card, if you have such a security feature enabled. Just like with any other mobile phone, you need to enter this every time you turn on the phone, although normally the iPhone is just asleep, not turned off. If you want to turn it off you have to hold down the sleep/power button for about 5 seconds and then confirm that you wish to power off. 
  4. **Get HTTP working through GPRS.** This is the step to get everything except Mail working &#8212; everything that uses http, that is: Safari, Maps, Stocks, and Weather. I found that YouTube did not work this way, nor did the iTunes, but I think that is because both work only through WiFi. Probably the 3G version will fix this&#8230; In the meantime, here are the steps you need. These are specific to Meteor Ireland. 
      * **Sign up for a data plan**. It is possible to enter the wap settings for EDGE and to use the Meteor proxy server (following similar instructions to the ones from a [similar page](http://web.taillard.net/iphonevoda "Get an iPhone working with Vodafone Ireland") explaining how to do this with Vodafone Ireland), but this method is punitively expensive! I signed up for Meteor&#8217;s most basic plan at â‚¬20/month and paid for a â‚¬9.99 data add-on that gives me 250MB per month. 
      * **Phone Customer Care** at 1905 and **make sure they turn on the isp networking** for you. This is the trickiest part since the Customer Care person may think you are talking about wap, which you are not, and since they must request to a technical support person that they activate this on their system for your account. Evidently by default it is not turned on. I had to make about five calls to Customer Care before Meteor got this right. 
      * **Enter the EDGE settings for Meteor**. These are:
  
        APN: <span style="color: #0000ff;">isp.mymeteor.ie</span>
  
        Username: <span style="color: #0000ff;">my</span>
  
        Password: <span style="color: #0000ff;">meteor
 </span>
      * Yes you do need to tur**n your phone off and turn it back on**. Before you do this, be sure to deactivate Wi-Fi if you have been using that, since otherwise you won&#8217;t be able to test the EDGE/GPRS function. Once you do that it should work! 
  5. **Things that still do not work:** 
      * **Voicemail.** Visual voicemail will probably never work, but there are ways to hack the phone so that the voicemail button calls Meteor voicemail. I am working on this too.

