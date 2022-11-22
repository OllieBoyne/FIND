var cur_vid = 0; // ID of video selected, [rgb, grey, chamf]


function show_vid(i) {
  var video_rgb = document.getElementById("video-rgb")
  var video_grey = document.getElementById("video-grey")
  var video_chamf = document.getElementById("video-chamf")
  
  
  videos = [video_rgb, video_grey, video_chamf]

  videos[i].pause()
  videos[i].currentTime = videos[cur_vid].currentTime
  videos[i].play()

  
  set_visibility = function(i) {
    video_rgb.style.display = ["block", "none", "none"][i]
    video_grey.style.display = ["none", "block", "none"][i]
    video_chamf.style.display = ["none", "none", "block"][i]
  }
  
  set_visibility(i)

  buttons = [document.getElementById("button-rgb"), document.getElementById("button-grey"),
              document.getElementById("button-chamf")]

  buttons[cur_vid].className = 'video_button'
  buttons[i].className = 'video_button_clicked'
  
  cur_vid = i

}

show_rgb = function() {show_vid(0)}
show_grey = function() {show_vid(1)}
show_chamf = function() {show_vid(2)}

// Need to load all videos in by running show_ for each
window.addEventListener('load', function() {show_rgb(); show_chamf(); show_grey()})