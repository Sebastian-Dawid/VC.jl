module ImageViewExt

using VC
import ImageView, Gtk4

function VC.imshow(::Nothing, img)
        guidict = ImageView.imshow(img)
        if !isinteractive()
            c = Condition()
            win = guidict["gui"]["window"]
            @async Gtk4.GLib.glib_main()
            Gtk4.signal_connect(win, :close_request) do widget
                Gtk4.notify(c)
            end
            Gtk4.wait(c)
	    Gtk4.close(win)
        end
end

function __init__()
    VC.IMAGEVIEW_LOADED = true
end

end
