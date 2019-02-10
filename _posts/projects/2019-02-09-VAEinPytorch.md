---
layout: project
permalink: /:title/
category: projects

meta:
  keywords: "VAE, pytorch"

project:
  title: "VAE in Pytorch"
  type: "Deep learning"
  url: "https://github.com/arnolds/pineapple"
  logo: "/assets/images/projects/aquapineapple/logo.png"
  tech: "PyTorch"

agency:
  url: "https://github.com/arnolds/pineapple"
  year: "2019"

images:
  - image:
    url: "/assets/images/projects/aquapineapple/devices.jpg"
    alt: "Aqua Pineapple website on tablet, mobile and desktop"
  - image:
    url: "/assets/images/projects/aquapineapple/desktop.jpg"
    alt: "Aqua Pineapple website on a desktop device"
  - image:
    url: "/assets/images/projects/aquapineapple/mobile.jpg"
    alt: "Aqua Pineapple website on a mobile device"
---

<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />
<title>VAE in numpy</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>

<style type="text/css">
    /*!
*
* Twitter Bootstrap
*
*/
/*!
 * Bootstrap v3.3.7 (http://getbootstrap.com)
 * Copyright 2011-2016 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
 */
/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type="button"],
input[type="reset"],
input[type="submit"] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type="checkbox"],
input[type="radio"] {
  box-sizing: border-box;
  padding: 0;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type="search"]::-webkit-search-cancel-button,
input[type="search"]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/*! Source: https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css */
@media print {
  *,
  *:before,
  *:after {
    background: transparent !important;
    color: #000 !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: " (" attr(href) ")";
  }
  abbr[title]:after {
    content: " (" attr(title) ")";
  }
  a[href^="#"]:after,
  a[href^="javascript:"]:after {
    content: "";
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn > .caret,
  .dropup > .btn > .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: 'Glyphicons Halflings';
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot');
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix') format('embedded-opentype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff2') format('woff2'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff') format('woff'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.ttf') format('truetype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular') format('svg');
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: 'Glyphicons Halflings';
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: "\002a";
}
.glyphicon-plus:before {
  content: "\002b";
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: "\20ac";
}
.glyphicon-minus:before {
  content: "\2212";
}
.glyphicon-cloud:before {
  content: "\2601";
}
.glyphicon-envelope:before {
  content: "\2709";
}
.glyphicon-pencil:before {
  content: "\270f";
}
.glyphicon-glass:before {
  content: "\e001";
}
.glyphicon-music:before {
  content: "\e002";
}
.glyphicon-search:before {
  content: "\e003";
}
.glyphicon-heart:before {
  content: "\e005";
}
.glyphicon-star:before {
  content: "\e006";
}
.glyphicon-star-empty:before {
  content: "\e007";
}
.glyphicon-user:before {
  content: "\e008";
}
.glyphicon-film:before {
  content: "\e009";
}
.glyphicon-th-large:before {
  content: "\e010";
}
.glyphicon-th:before {
  content: "\e011";
}
.glyphicon-th-list:before {
  content: "\e012";
}
.glyphicon-ok:before {
  content: "\e013";
}
.glyphicon-remove:before {
  content: "\e014";
}
.glyphicon-zoom-in:before {
  content: "\e015";
}
.glyphicon-zoom-out:before {
  content: "\e016";
}
.glyphicon-off:before {
  content: "\e017";
}
.glyphicon-signal:before {
  content: "\e018";
}
.glyphicon-cog:before {
  content: "\e019";
}
.glyphicon-trash:before {
  content: "\e020";
}
.glyphicon-home:before {
  content: "\e021";
}
.glyphicon-file:before {
  content: "\e022";
}
.glyphicon-time:before {
  content: "\e023";
}
.glyphicon-road:before {
  content: "\e024";
}
.glyphicon-download-alt:before {
  content: "\e025";
}
.glyphicon-download:before {
  content: "\e026";
}
.glyphicon-upload:before {
  content: "\e027";
}
.glyphicon-inbox:before {
  content: "\e028";
}
.glyphicon-play-circle:before {
  content: "\e029";
}
.glyphicon-repeat:before {
  content: "\e030";
}
.glyphicon-refresh:before {
  content: "\e031";
}
.glyphicon-list-alt:before {
  content: "\e032";
}
.glyphicon-lock:before {
  content: "\e033";
}
.glyphicon-flag:before {
  content: "\e034";
}
.glyphicon-headphones:before {
  content: "\e035";
}
.glyphicon-volume-off:before {
  content: "\e036";
}
.glyphicon-volume-down:before {
  content: "\e037";
}
.glyphicon-volume-up:before {
  content: "\e038";
}
.glyphicon-qrcode:before {
  content: "\e039";
}
.glyphicon-barcode:before {
  content: "\e040";
}
.glyphicon-tag:before {
  content: "\e041";
}
.glyphicon-tags:before {
  content: "\e042";
}
.glyphicon-book:before {
  content: "\e043";
}
.glyphicon-bookmark:before {
  content: "\e044";
}
.glyphicon-print:before {
  content: "\e045";
}
.glyphicon-camera:before {
  content: "\e046";
}
.glyphicon-font:before {
  content: "\e047";
}
.glyphicon-bold:before {
  content: "\e048";
}
.glyphicon-italic:before {
  content: "\e049";
}
.glyphicon-text-height:before {
  content: "\e050";
}
.glyphicon-text-width:before {
  content: "\e051";
}
.glyphicon-align-left:before {
  content: "\e052";
}
.glyphicon-align-center:before {
  content: "\e053";
}
.glyphicon-align-right:before {
  content: "\e054";
}
.glyphicon-align-justify:before {
  content: "\e055";
}
.glyphicon-list:before {
  content: "\e056";
}
.glyphicon-indent-left:before {
  content: "\e057";
}
.glyphicon-indent-right:before {
  content: "\e058";
}
.glyphicon-facetime-video:before {
  content: "\e059";
}
.glyphicon-picture:before {
  content: "\e060";
}
.glyphicon-map-marker:before {
  content: "\e062";
}
.glyphicon-adjust:before {
  content: "\e063";
}
.glyphicon-tint:before {
  content: "\e064";
}
.glyphicon-edit:before {
  content: "\e065";
}
.glyphicon-share:before {
  content: "\e066";
}
.glyphicon-check:before {
  content: "\e067";
}
.glyphicon-move:before {
  content: "\e068";
}
.glyphicon-step-backward:before {
  content: "\e069";
}
.glyphicon-fast-backward:before {
  content: "\e070";
}
.glyphicon-backward:before {
  content: "\e071";
}
.glyphicon-play:before {
  content: "\e072";
}
.glyphicon-pause:before {
  content: "\e073";
}
.glyphicon-stop:before {
  content: "\e074";
}
.glyphicon-forward:before {
  content: "\e075";
}
.glyphicon-fast-forward:before {
  content: "\e076";
}
.glyphicon-step-forward:before {
  content: "\e077";
}
.glyphicon-eject:before {
  content: "\e078";
}
.glyphicon-chevron-left:before {
  content: "\e079";
}
.glyphicon-chevron-right:before {
  content: "\e080";
}
.glyphicon-plus-sign:before {
  content: "\e081";
}
.glyphicon-minus-sign:before {
  content: "\e082";
}
.glyphicon-remove-sign:before {
  content: "\e083";
}
.glyphicon-ok-sign:before {
  content: "\e084";
}
.glyphicon-question-sign:before {
  content: "\e085";
}
.glyphicon-info-sign:before {
  content: "\e086";
}
.glyphicon-screenshot:before {
  content: "\e087";
}
.glyphicon-remove-circle:before {
  content: "\e088";
}
.glyphicon-ok-circle:before {
  content: "\e089";
}
.glyphicon-ban-circle:before {
  content: "\e090";
}
.glyphicon-arrow-left:before {
  content: "\e091";
}
.glyphicon-arrow-right:before {
  content: "\e092";
}
.glyphicon-arrow-up:before {
  content: "\e093";
}
.glyphicon-arrow-down:before {
  content: "\e094";
}
.glyphicon-share-alt:before {
  content: "\e095";
}
.glyphicon-resize-full:before {
  content: "\e096";
}
.glyphicon-resize-small:before {
  content: "\e097";
}
.glyphicon-exclamation-sign:before {
  content: "\e101";
}
.glyphicon-gift:before {
  content: "\e102";
}
.glyphicon-leaf:before {
  content: "\e103";
}
.glyphicon-fire:before {
  content: "\e104";
}
.glyphicon-eye-open:before {
  content: "\e105";
}
.glyphicon-eye-close:before {
  content: "\e106";
}
.glyphicon-warning-sign:before {
  content: "\e107";
}
.glyphicon-plane:before {
  content: "\e108";
}
.glyphicon-calendar:before {
  content: "\e109";
}
.glyphicon-random:before {
  content: "\e110";
}
.glyphicon-comment:before {
  content: "\e111";
}
.glyphicon-magnet:before {
  content: "\e112";
}
.glyphicon-chevron-up:before {
  content: "\e113";
}
.glyphicon-chevron-down:before {
  content: "\e114";
}
.glyphicon-retweet:before {
  content: "\e115";
}
.glyphicon-shopping-cart:before {
  content: "\e116";
}
.glyphicon-folder-close:before {
  content: "\e117";
}
.glyphicon-folder-open:before {
  content: "\e118";
}
.glyphicon-resize-vertical:before {
  content: "\e119";
}
.glyphicon-resize-horizontal:before {
  content: "\e120";
}
.glyphicon-hdd:before {
  content: "\e121";
}
.glyphicon-bullhorn:before {
  content: "\e122";
}
.glyphicon-bell:before {
  content: "\e123";
}
.glyphicon-certificate:before {
  content: "\e124";
}
.glyphicon-thumbs-up:before {
  content: "\e125";
}
.glyphicon-thumbs-down:before {
  content: "\e126";
}
.glyphicon-hand-right:before {
  content: "\e127";
}
.glyphicon-hand-left:before {
  content: "\e128";
}
.glyphicon-hand-up:before {
  content: "\e129";
}
.glyphicon-hand-down:before {
  content: "\e130";
}
.glyphicon-circle-arrow-right:before {
  content: "\e131";
}
.glyphicon-circle-arrow-left:before {
  content: "\e132";
}
.glyphicon-circle-arrow-up:before {
  content: "\e133";
}
.glyphicon-circle-arrow-down:before {
  content: "\e134";
}
.glyphicon-globe:before {
  content: "\e135";
}
.glyphicon-wrench:before {
  content: "\e136";
}
.glyphicon-tasks:before {
  content: "\e137";
}
.glyphicon-filter:before {
  content: "\e138";
}
.glyphicon-briefcase:before {
  content: "\e139";
}
.glyphicon-fullscreen:before {
  content: "\e140";
}
.glyphicon-dashboard:before {
  content: "\e141";
}
.glyphicon-paperclip:before {
  content: "\e142";
}
.glyphicon-heart-empty:before {
  content: "\e143";
}
.glyphicon-link:before {
  content: "\e144";
}
.glyphicon-phone:before {
  content: "\e145";
}
.glyphicon-pushpin:before {
  content: "\e146";
}
.glyphicon-usd:before {
  content: "\e148";
}
.glyphicon-gbp:before {
  content: "\e149";
}
.glyphicon-sort:before {
  content: "\e150";
}
.glyphicon-sort-by-alphabet:before {
  content: "\e151";
}
.glyphicon-sort-by-alphabet-alt:before {
  content: "\e152";
}
.glyphicon-sort-by-order:before {
  content: "\e153";
}
.glyphicon-sort-by-order-alt:before {
  content: "\e154";
}
.glyphicon-sort-by-attributes:before {
  content: "\e155";
}
.glyphicon-sort-by-attributes-alt:before {
  content: "\e156";
}
.glyphicon-unchecked:before {
  content: "\e157";
}
.glyphicon-expand:before {
  content: "\e158";
}
.glyphicon-collapse-down:before {
  content: "\e159";
}
.glyphicon-collapse-up:before {
  content: "\e160";
}
.glyphicon-log-in:before {
  content: "\e161";
}
.glyphicon-flash:before {
  content: "\e162";
}
.glyphicon-log-out:before {
  content: "\e163";
}
.glyphicon-new-window:before {
  content: "\e164";
}
.glyphicon-record:before {
  content: "\e165";
}
.glyphicon-save:before {
  content: "\e166";
}
.glyphicon-open:before {
  content: "\e167";
}
.glyphicon-saved:before {
  content: "\e168";
}
.glyphicon-import:before {
  content: "\e169";
}
.glyphicon-export:before {
  content: "\e170";
}
.glyphicon-send:before {
  content: "\e171";
}
.glyphicon-floppy-disk:before {
  content: "\e172";
}
.glyphicon-floppy-saved:before {
  content: "\e173";
}
.glyphicon-floppy-remove:before {
  content: "\e174";
}
.glyphicon-floppy-save:before {
  content: "\e175";
}
.glyphicon-floppy-open:before {
  content: "\e176";
}
.glyphicon-credit-card:before {
  content: "\e177";
}
.glyphicon-transfer:before {
  content: "\e178";
}
.glyphicon-cutlery:before {
  content: "\e179";
}
.glyphicon-header:before {
  content: "\e180";
}
.glyphicon-compressed:before {
  content: "\e181";
}
.glyphicon-earphone:before {
  content: "\e182";
}
.glyphicon-phone-alt:before {
  content: "\e183";
}
.glyphicon-tower:before {
  content: "\e184";
}
.glyphicon-stats:before {
  content: "\e185";
}
.glyphicon-sd-video:before {
  content: "\e186";
}
.glyphicon-hd-video:before {
  content: "\e187";
}
.glyphicon-subtitles:before {
  content: "\e188";
}
.glyphicon-sound-stereo:before {
  content: "\e189";
}
.glyphicon-sound-dolby:before {
  content: "\e190";
}
.glyphicon-sound-5-1:before {
  content: "\e191";
}
.glyphicon-sound-6-1:before {
  content: "\e192";
}
.glyphicon-sound-7-1:before {
  content: "\e193";
}
.glyphicon-copyright-mark:before {
  content: "\e194";
}
.glyphicon-registration-mark:before {
  content: "\e195";
}
.glyphicon-cloud-download:before {
  content: "\e197";
}
.glyphicon-cloud-upload:before {
  content: "\e198";
}
.glyphicon-tree-conifer:before {
  content: "\e199";
}
.glyphicon-tree-deciduous:before {
  content: "\e200";
}
.glyphicon-cd:before {
  content: "\e201";
}
.glyphicon-save-file:before {
  content: "\e202";
}
.glyphicon-open-file:before {
  content: "\e203";
}
.glyphicon-level-up:before {
  content: "\e204";
}
.glyphicon-copy:before {
  content: "\e205";
}
.glyphicon-paste:before {
  content: "\e206";
}
.glyphicon-alert:before {
  content: "\e209";
}
.glyphicon-equalizer:before {
  content: "\e210";
}
.glyphicon-king:before {
  content: "\e211";
}
.glyphicon-queen:before {
  content: "\e212";
}
.glyphicon-pawn:before {
  content: "\e213";
}
.glyphicon-bishop:before {
  content: "\e214";
}
.glyphicon-knight:before {
  content: "\e215";
}
.glyphicon-baby-formula:before {
  content: "\e216";
}
.glyphicon-tent:before {
  content: "\26fa";
}
.glyphicon-blackboard:before {
  content: "\e218";
}
.glyphicon-bed:before {
  content: "\e219";
}
.glyphicon-apple:before {
  content: "\f8ff";
}
.glyphicon-erase:before {
  content: "\e221";
}
.glyphicon-hourglass:before {
  content: "\231b";
}
.glyphicon-lamp:before {
  content: "\e223";
}
.glyphicon-duplicate:before {
  content: "\e224";
}
.glyphicon-piggy-bank:before {
  content: "\e225";
}
.glyphicon-scissors:before {
  content: "\e226";
}
.glyphicon-bitcoin:before {
  content: "\e227";
}
.glyphicon-btc:before {
  content: "\e227";
}
.glyphicon-xbt:before {
  content: "\e227";
}
.glyphicon-yen:before {
  content: "\00a5";
}
.glyphicon-jpy:before {
  content: "\00a5";
}
.glyphicon-ruble:before {
  content: "\20bd";
}
.glyphicon-rub:before {
  content: "\20bd";
}
.glyphicon-scale:before {
  content: "\e230";
}
.glyphicon-ice-lolly:before {
  content: "\e231";
}
.glyphicon-ice-lolly-tasted:before {
  content: "\e232";
}
.glyphicon-education:before {
  content: "\e233";
}
.glyphicon-option-horizontal:before {
  content: "\e234";
}
.glyphicon-option-vertical:before {
  content: "\e235";
}
.glyphicon-menu-hamburger:before {
  content: "\e236";
}
.glyphicon-modal-window:before {
  content: "\e237";
}
.glyphicon-oil:before {
  content: "\e238";
}
.glyphicon-grain:before {
  content: "\e239";
}
.glyphicon-sunglasses:before {
  content: "\e240";
}
.glyphicon-text-size:before {
  content: "\e241";
}
.glyphicon-text-color:before {
  content: "\e242";
}
.glyphicon-text-background:before {
  content: "\e243";
}
.glyphicon-object-align-top:before {
  content: "\e244";
}
.glyphicon-object-align-bottom:before {
  content: "\e245";
}
.glyphicon-object-align-horizontal:before {
  content: "\e246";
}
.glyphicon-object-align-left:before {
  content: "\e247";
}
.glyphicon-object-align-vertical:before {
  content: "\e248";
}
.glyphicon-object-align-right:before {
  content: "\e249";
}
.glyphicon-triangle-right:before {
  content: "\e250";
}
.glyphicon-triangle-left:before {
  content: "\e251";
}
.glyphicon-triangle-bottom:before {
  content: "\e252";
}
.glyphicon-triangle-top:before {
  content: "\e253";
}
.glyphicon-console:before {
  content: "\e254";
}
.glyphicon-superscript:before {
  content: "\e255";
}
.glyphicon-subscript:before {
  content: "\e256";
}
.glyphicon-menu-left:before {
  content: "\e257";
}
.glyphicon-menu-right:before {
  content: "\e258";
}
.glyphicon-menu-down:before {
  content: "\e259";
}
.glyphicon-menu-up:before {
  content: "\e260";
}
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
*:before,
*:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail > img,
.thumbnail a > img,
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role="button"] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline > li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: '\2014 \00A0';
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: '';
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: '\00A0 \2014';
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table > thead > tr > th,
.table > tbody > tr > th,
.table > tfoot > tr > th,
.table > thead > tr > td,
.table > tbody > tr > td,
.table > tfoot > tr > td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table > thead > tr > th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table > caption + thead > tr:first-child > th,
.table > colgroup + thead > tr:first-child > th,
.table > thead:first-child > tr:first-child > th,
.table > caption + thead > tr:first-child > td,
.table > colgroup + thead > tr:first-child > td,
.table > thead:first-child > tr:first-child > td {
  border-top: 0;
}
.table > tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed > thead > tr > th,
.table-condensed > tbody > tr > th,
.table-condensed > tfoot > tr > th,
.table-condensed > thead > tr > td,
.table-condensed > tbody > tr > td,
.table-condensed > tfoot > tr > td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > tbody > tr > th,
.table-bordered > tfoot > tr > th,
.table-bordered > thead > tr > td,
.table-bordered > tbody > tr > td,
.table-bordered > tfoot > tr > td {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > thead > tr > td {
  border-bottom-width: 2px;
}
.table-striped > tbody > tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover > tbody > tr:hover {
  background-color: #f5f5f5;
}
table col[class*="col-"] {
  position: static;
  float: none;
  display: table-column;
}
table td[class*="col-"],
table th[class*="col-"] {
  position: static;
  float: none;
  display: table-cell;
}
.table > thead > tr > td.active,
.table > tbody > tr > td.active,
.table > tfoot > tr > td.active,
.table > thead > tr > th.active,
.table > tbody > tr > th.active,
.table > tfoot > tr > th.active,
.table > thead > tr.active > td,
.table > tbody > tr.active > td,
.table > tfoot > tr.active > td,
.table > thead > tr.active > th,
.table > tbody > tr.active > th,
.table > tfoot > tr.active > th {
  background-color: #f5f5f5;
}
.table-hover > tbody > tr > td.active:hover,
.table-hover > tbody > tr > th.active:hover,
.table-hover > tbody > tr.active:hover > td,
.table-hover > tbody > tr:hover > .active,
.table-hover > tbody > tr.active:hover > th {
  background-color: #e8e8e8;
}
.table > thead > tr > td.success,
.table > tbody > tr > td.success,
.table > tfoot > tr > td.success,
.table > thead > tr > th.success,
.table > tbody > tr > th.success,
.table > tfoot > tr > th.success,
.table > thead > tr.success > td,
.table > tbody > tr.success > td,
.table > tfoot > tr.success > td,
.table > thead > tr.success > th,
.table > tbody > tr.success > th,
.table > tfoot > tr.success > th {
  background-color: #dff0d8;
}
.table-hover > tbody > tr > td.success:hover,
.table-hover > tbody > tr > th.success:hover,
.table-hover > tbody > tr.success:hover > td,
.table-hover > tbody > tr:hover > .success,
.table-hover > tbody > tr.success:hover > th {
  background-color: #d0e9c6;
}
.table > thead > tr > td.info,
.table > tbody > tr > td.info,
.table > tfoot > tr > td.info,
.table > thead > tr > th.info,
.table > tbody > tr > th.info,
.table > tfoot > tr > th.info,
.table > thead > tr.info > td,
.table > tbody > tr.info > td,
.table > tfoot > tr.info > td,
.table > thead > tr.info > th,
.table > tbody > tr.info > th,
.table > tfoot > tr.info > th {
  background-color: #d9edf7;
}
.table-hover > tbody > tr > td.info:hover,
.table-hover > tbody > tr > th.info:hover,
.table-hover > tbody > tr.info:hover > td,
.table-hover > tbody > tr:hover > .info,
.table-hover > tbody > tr.info:hover > th {
  background-color: #c4e3f3;
}
.table > thead > tr > td.warning,
.table > tbody > tr > td.warning,
.table > tfoot > tr > td.warning,
.table > thead > tr > th.warning,
.table > tbody > tr > th.warning,
.table > tfoot > tr > th.warning,
.table > thead > tr.warning > td,
.table > tbody > tr.warning > td,
.table > tfoot > tr.warning > td,
.table > thead > tr.warning > th,
.table > tbody > tr.warning > th,
.table > tfoot > tr.warning > th {
  background-color: #fcf8e3;
}
.table-hover > tbody > tr > td.warning:hover,
.table-hover > tbody > tr > th.warning:hover,
.table-hover > tbody > tr.warning:hover > td,
.table-hover > tbody > tr:hover > .warning,
.table-hover > tbody > tr.warning:hover > th {
  background-color: #faf2cc;
}
.table > thead > tr > td.danger,
.table > tbody > tr > td.danger,
.table > tfoot > tr > td.danger,
.table > thead > tr > th.danger,
.table > tbody > tr > th.danger,
.table > tfoot > tr > th.danger,
.table > thead > tr.danger > td,
.table > tbody > tr.danger > td,
.table > tfoot > tr.danger > td,
.table > thead > tr.danger > th,
.table > tbody > tr.danger > th,
.table > tfoot > tr.danger > th {
  background-color: #f2dede;
}
.table-hover > tbody > tr > td.danger:hover,
.table-hover > tbody > tr > th.danger:hover,
.table-hover > tbody > tr.danger:hover > td,
.table-hover > tbody > tr:hover > .danger,
.table-hover > tbody > tr.danger:hover > th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive > .table {
    margin-bottom: 0;
  }
  .table-responsive > .table > thead > tr > th,
  .table-responsive > .table > tbody > tr > th,
  .table-responsive > .table > tfoot > tr > th,
  .table-responsive > .table > thead > tr > td,
  .table-responsive > .table > tbody > tr > td,
  .table-responsive > .table > tfoot > tr > td {
    white-space: nowrap;
  }
  .table-responsive > .table-bordered {
    border: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:first-child,
  .table-responsive > .table-bordered > tbody > tr > th:first-child,
  .table-responsive > .table-bordered > tfoot > tr > th:first-child,
  .table-responsive > .table-bordered > thead > tr > td:first-child,
  .table-responsive > .table-bordered > tbody > tr > td:first-child,
  .table-responsive > .table-bordered > tfoot > tr > td:first-child {
    border-left: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:last-child,
  .table-responsive > .table-bordered > tbody > tr > th:last-child,
  .table-responsive > .table-bordered > tfoot > tr > th:last-child,
  .table-responsive > .table-bordered > thead > tr > td:last-child,
  .table-responsive > .table-bordered > tbody > tr > td:last-child,
  .table-responsive > .table-bordered > tfoot > tr > td:last-child {
    border-right: 0;
  }
  .table-responsive > .table-bordered > tbody > tr:last-child > th,
  .table-responsive > .table-bordered > tfoot > tr:last-child > th,
  .table-responsive > .table-bordered > tbody > tr:last-child > td,
  .table-responsive > .table-bordered > tfoot > tr:last-child > td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="search"] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type="radio"],
input[type="checkbox"] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type="file"] {
  display: block;
}
input[type="range"] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type="file"]:focus,
input[type="radio"]:focus,
input[type="checkbox"]:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type="date"].form-control,
  input[type="time"].form-control,
  input[type="datetime-local"].form-control,
  input[type="month"].form-control {
    line-height: 32px;
  }
  input[type="date"].input-sm,
  input[type="time"].input-sm,
  input[type="datetime-local"].input-sm,
  input[type="month"].input-sm,
  .input-group-sm input[type="date"],
  .input-group-sm input[type="time"],
  .input-group-sm input[type="datetime-local"],
  .input-group-sm input[type="month"] {
    line-height: 30px;
  }
  input[type="date"].input-lg,
  input[type="time"].input-lg,
  input[type="datetime-local"].input-lg,
  input[type="month"].input-lg,
  .input-group-lg input[type="date"],
  .input-group-lg input[type="time"],
  .input-group-lg input[type="datetime-local"],
  .input-group-lg input[type="month"] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type="radio"],
.radio-inline input[type="radio"],
.checkbox input[type="checkbox"],
.checkbox-inline input[type="checkbox"] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type="radio"][disabled],
input[type="checkbox"][disabled],
input[type="radio"].disabled,
input[type="checkbox"].disabled,
fieldset[disabled] input[type="radio"],
fieldset[disabled] input[type="checkbox"] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group > .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type="radio"],
  .form-inline .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open > .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open > .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open > .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open > .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open > .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open > .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open > .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open > .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open > .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open > .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open > .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open > .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open > .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open > .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open > .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open > .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open > .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open > .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg > .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm > .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs > .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type="submit"].btn-block,
input[type="reset"].btn-block,
input[type="button"].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu > li > a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu > li > a:hover,
.dropdown-menu > li > a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu > .active > a,
.dropdown-menu > .active > a:hover,
.dropdown-menu > .active > a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu > .disabled > a,
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  color: #777777;
}
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open > .dropdown-menu {
  display: block;
}
.open > a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right > .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: "";
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  float: left;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover,
.btn-group > .btn:focus,
.btn-group-vertical > .btn:focus,
.btn-group > .btn:active,
.btn-group-vertical > .btn:active,
.btn-group > .btn.active,
.btn-group-vertical > .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar > .btn,
.btn-toolbar > .btn-group,
.btn-toolbar > .input-group {
  margin-left: 5px;
}
.btn-group > .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group > .btn:first-child {
  margin-left: 0;
}
.btn-group > .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn:last-child:not(:first-child),
.btn-group > .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group > .btn-group {
  float: left;
}
.btn-group > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group > .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group > .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group,
.btn-group-vertical > .btn-group > .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical > .btn-group > .btn {
  float: none;
}
.btn-group-vertical > .btn + .btn,
.btn-group-vertical > .btn + .btn-group,
.btn-group-vertical > .btn-group + .btn,
.btn-group-vertical > .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical > .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical > .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified > .btn,
.btn-group-justified > .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified > .btn-group .btn {
  width: 100%;
}
.btn-group-justified > .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle="buttons"] > .btn input[type="radio"],
[data-toggle="buttons"] > .btn-group > .btn input[type="radio"],
[data-toggle="buttons"] > .btn input[type="checkbox"],
[data-toggle="buttons"] > .btn-group > .btn input[type="checkbox"] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class*="col-"] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg > .form-control,
.input-group-lg > .input-group-addon,
.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg > .form-control,
select.input-group-lg > .input-group-addon,
select.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg > .form-control,
textarea.input-group-lg > .input-group-addon,
textarea.input-group-lg > .input-group-btn > .btn,
select[multiple].input-group-lg > .form-control,
select[multiple].input-group-lg > .input-group-addon,
select[multiple].input-group-lg > .input-group-btn > .btn {
  height: auto;
}
.input-group-sm > .form-control,
.input-group-sm > .input-group-addon,
.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm > .form-control,
select.input-group-sm > .input-group-addon,
select.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm > .form-control,
textarea.input-group-sm > .input-group-addon,
textarea.input-group-sm > .input-group-btn > .btn,
select[multiple].input-group-sm > .form-control,
select[multiple].input-group-sm > .input-group-addon,
select[multiple].input-group-sm > .input-group-btn > .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type="radio"],
.input-group-addon input[type="checkbox"] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group > .btn,
.input-group-btn:first-child > .dropdown-toggle,
.input-group-btn:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group > .btn,
.input-group-btn:last-child > .dropdown-toggle,
.input-group-btn:first-child > .btn:not(:first-child),
.input-group-btn:first-child > .btn-group:not(:first-child) > .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn > .btn {
  position: relative;
}
.input-group-btn > .btn + .btn {
  margin-left: -1px;
}
.input-group-btn > .btn:hover,
.input-group-btn > .btn:focus,
.input-group-btn > .btn:active {
  z-index: 2;
}
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav > li {
  position: relative;
  display: block;
}
.nav > li > a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav > li > a:hover,
.nav > li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav > li.disabled > a {
  color: #777777;
}
.nav > li.disabled > a:hover,
.nav > li.disabled > a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open > a,
.nav .open > a:hover,
.nav .open > a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav > li > a > img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs > li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs > li > a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs > li > a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs > li.active > a,
.nav-tabs > li.active > a:hover,
.nav-tabs > li.active > a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified > li {
  float: none;
}
.nav-tabs.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified > .active > a,
.nav-tabs.nav-justified > .active > a:hover,
.nav-tabs.nav-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified > .active > a,
  .nav-tabs.nav-justified > .active > a:hover,
  .nav-tabs.nav-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills > li {
  float: left;
}
.nav-pills > li > a {
  border-radius: 2px;
}
.nav-pills > li + li {
  margin-left: 2px;
}
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked > li {
  float: none;
}
.nav-stacked > li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified > li {
  float: none;
}
.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified > .active > a,
.nav-tabs-justified > .active > a:hover,
.nav-tabs-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified > .active > a,
  .nav-tabs-justified > .active > a:hover,
  .nav-tabs-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container > .navbar-header,
.container-fluid > .navbar-header,
.container > .navbar-collapse,
.container-fluid > .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container > .navbar-header,
  .container-fluid > .navbar-header,
  .container > .navbar-collapse,
  .container-fluid > .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand > img {
  display: block;
}
@media (min-width: 541px) {
  .navbar > .container .navbar-brand,
  .navbar > .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav > li > a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu > li > a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu > li > a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-nav .open .dropdown-menu > li > a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav > li {
    float: left;
  }
  .navbar-nav > li > a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group > .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type="radio"],
  .navbar-form .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav > li > .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav > li > .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav > li > a {
  color: #777;
}
.navbar-default .navbar-nav > li > a:hover,
.navbar-default .navbar-nav > li > a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav > .active > a,
.navbar-default .navbar-nav > .active > a:hover,
.navbar-default .navbar-nav > .active > a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav > .disabled > a,
.navbar-default .navbar-nav > .disabled > a:hover,
.navbar-default .navbar-nav > .disabled > a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav > .open > a,
.navbar-default .navbar-nav > .open > a:hover,
.navbar-default .navbar-nav > .open > a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu > li > a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a:hover,
.navbar-inverse .navbar-nav > li > a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav > .active > a,
.navbar-inverse .navbar-nav > .active > a:hover,
.navbar-inverse .navbar-nav > .active > a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav > .disabled > a,
.navbar-inverse .navbar-nav > .disabled > a:hover,
.navbar-inverse .navbar-nav > .disabled > a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav > .open > a,
.navbar-inverse .navbar-nav > .open > a:hover,
.navbar-inverse .navbar-nav > .open > a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu > .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb > li {
  display: inline-block;
}
.breadcrumb > li + li:before {
  content: "/\00a0";
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb > .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination > li {
  display: inline;
}
.pagination > li > a,
.pagination > li > span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination > li:first-child > a,
.pagination > li:first-child > span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination > li:last-child > a,
.pagination > li:last-child > span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination > li > a:hover,
.pagination > li > span:hover,
.pagination > li > a:focus,
.pagination > li > span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination > .active > a,
.pagination > .active > span,
.pagination > .active > a:hover,
.pagination > .active > span:hover,
.pagination > .active > a:focus,
.pagination > .active > span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination > .disabled > span,
.pagination > .disabled > span:hover,
.pagination > .disabled > span:focus,
.pagination > .disabled > a,
.pagination > .disabled > a:hover,
.pagination > .disabled > a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg > li > a,
.pagination-lg > li > span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg > li:first-child > a,
.pagination-lg > li:first-child > span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg > li:last-child > a,
.pagination-lg > li:last-child > span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm > li > a,
.pagination-sm > li > span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm > li:first-child > a,
.pagination-sm > li:first-child > span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm > li:last-child > a,
.pagination-sm > li:last-child > span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li > a,
.pager li > span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li > a:hover,
.pager li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next > a,
.pager .next > span {
  float: right;
}
.pager .previous > a,
.pager .previous > span {
  float: left;
}
.pager .disabled > a,
.pager .disabled > a:hover,
.pager .disabled > a:focus,
.pager .disabled > span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs > .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active > .badge,
.nav-pills > .active > a > .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item > .badge {
  float: right;
}
.list-group-item > .badge + .badge {
  margin-right: 5px;
}
.nav-pills > li > a > .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron > hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail > img,
.thumbnail a > img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert > p,
.alert > ul {
  margin-bottom: 0;
}
.alert > p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media > .pull-right {
  padding-left: 10px;
}
.media-left,
.media > .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading > small,
.list-group-item.active:hover .list-group-item-heading > small,
.list-group-item.active:focus .list-group-item-heading > small,
.list-group-item.active .list-group-item-heading > .small,
.list-group-item.active:hover .list-group-item-heading > .small,
.list-group-item.active:focus .list-group-item-heading > .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading > .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title > a,
.panel-title > small,
.panel-title > .small,
.panel-title > small > a,
.panel-title > .small > a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .list-group,
.panel > .panel-collapse > .list-group {
  margin-bottom: 0;
}
.panel > .list-group .list-group-item,
.panel > .panel-collapse > .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel > .list-group:first-child .list-group-item:first-child,
.panel > .panel-collapse > .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .list-group:last-child .list-group-item:last-child,
.panel > .panel-collapse > .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .panel-heading + .panel-collapse > .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel > .table,
.panel > .table-responsive > .table,
.panel > .panel-collapse > .table {
  margin-bottom: 0;
}
.panel > .table caption,
.panel > .table-responsive > .table caption,
.panel > .panel-collapse > .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel > .table:first-child,
.panel > .table-responsive:first-child > .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel > .table:last-child,
.panel > .table-responsive:last-child > .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel > .panel-body + .table,
.panel > .panel-body + .table-responsive,
.panel > .table + .panel-body,
.panel > .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel > .table > tbody:first-child > tr:first-child th,
.panel > .table > tbody:first-child > tr:first-child td {
  border-top: 0;
}
.panel > .table-bordered,
.panel > .table-responsive > .table-bordered {
  border: 0;
}
.panel > .table-bordered > thead > tr > th:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:first-child,
.panel > .table-bordered > tbody > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:first-child,
.panel > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-bordered > thead > tr > td:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:first-child,
.panel > .table-bordered > tbody > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:first-child,
.panel > .table-bordered > tfoot > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:first-child {
  border-left: 0;
}
.panel > .table-bordered > thead > tr > th:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:last-child,
.panel > .table-bordered > tbody > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:last-child,
.panel > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-bordered > thead > tr > td:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:last-child,
.panel > .table-bordered > tbody > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:last-child,
.panel > .table-bordered > tfoot > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:last-child {
  border-right: 0;
}
.panel > .table-bordered > thead > tr:first-child > td,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > td,
.panel > .table-bordered > tbody > tr:first-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > td,
.panel > .table-bordered > thead > tr:first-child > th,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > th,
.panel > .table-bordered > tbody > tr:first-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > th {
  border-bottom: 0;
}
.panel > .table-bordered > tbody > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > td,
.panel > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-bordered > tbody > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > th,
.panel > .table-bordered > tfoot > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > th {
  border-bottom: 0;
}
.panel > .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse > .panel-body,
.panel-group .panel-heading + .panel-collapse > .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default > .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ddd;
}
.panel-default > .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary > .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #337ab7;
}
.panel-primary > .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success > .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success > .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info > .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #bce8f1;
}
.panel-info > .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning > .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #faebcc;
}
.panel-warning > .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger > .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger > .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover > .arrow,
.popover > .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover > .arrow {
  border-width: 11px;
}
.popover > .arrow:after {
  border-width: 10px;
  content: "";
}
.popover.top > .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top > .arrow:after {
  content: " ";
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right > .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right > .arrow:after {
  content: " ";
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom > .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom > .arrow:after {
  content: " ";
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left > .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left > .arrow:after {
  content: " ";
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner > .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner > .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner > .item.next,
  .carousel-inner > .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.prev,
  .carousel-inner > .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.next.left,
  .carousel-inner > .item.prev.right,
  .carousel-inner > .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner > .active,
.carousel-inner > .next,
.carousel-inner > .prev {
  display: block;
}
.carousel-inner > .active {
  left: 0;
}
.carousel-inner > .next,
.carousel-inner > .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner > .next {
  left: 100%;
}
.carousel-inner > .prev {
  left: -100%;
}
.carousel-inner > .next.left,
.carousel-inner > .prev.right {
  left: 0;
}
.carousel-inner > .active.left {
  left: -100%;
}
.carousel-inner > .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#80000000', endColorstr='#00000000', GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#80000000', GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: '\2039';
}
.carousel-control .icon-next:before {
  content: '\203a';
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:before,
.btn-group-vertical > .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: " ";
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/*!
*
* Font Awesome
*
*/
/*!
 *  Font Awesome 4.2.0 by @davegandy - http://fontawesome.io - @fontawesome
 *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)
 */
/* FONT PATH
 * -------------------------- */
@font-face {
  font-family: 'FontAwesome';
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.2.0');
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&v=4.2.0') format('embedded-opentype'), url('../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.2.0') format('woff'), url('../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.2.0') format('truetype'), url('../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.2.0#fontawesomeregular') format('svg');
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* makes the font 33% larger relative to the icon container */
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul > li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=1);
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2);
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=3);
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1);
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1);
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/* Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons */
.fa-glass:before {
  content: "\f000";
}
.fa-music:before {
  content: "\f001";
}
.fa-search:before {
  content: "\f002";
}
.fa-envelope-o:before {
  content: "\f003";
}
.fa-heart:before {
  content: "\f004";
}
.fa-star:before {
  content: "\f005";
}
.fa-star-o:before {
  content: "\f006";
}
.fa-user:before {
  content: "\f007";
}
.fa-film:before {
  content: "\f008";
}
.fa-th-large:before {
  content: "\f009";
}
.fa-th:before {
  content: "\f00a";
}
.fa-th-list:before {
  content: "\f00b";
}
.fa-check:before {
  content: "\f00c";
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: "\f00d";
}
.fa-search-plus:before {
  content: "\f00e";
}
.fa-search-minus:before {
  content: "\f010";
}
.fa-power-off:before {
  content: "\f011";
}
.fa-signal:before {
  content: "\f012";
}
.fa-gear:before,
.fa-cog:before {
  content: "\f013";
}
.fa-trash-o:before {
  content: "\f014";
}
.fa-home:before {
  content: "\f015";
}
.fa-file-o:before {
  content: "\f016";
}
.fa-clock-o:before {
  content: "\f017";
}
.fa-road:before {
  content: "\f018";
}
.fa-download:before {
  content: "\f019";
}
.fa-arrow-circle-o-down:before {
  content: "\f01a";
}
.fa-arrow-circle-o-up:before {
  content: "\f01b";
}
.fa-inbox:before {
  content: "\f01c";
}
.fa-play-circle-o:before {
  content: "\f01d";
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: "\f01e";
}
.fa-refresh:before {
  content: "\f021";
}
.fa-list-alt:before {
  content: "\f022";
}
.fa-lock:before {
  content: "\f023";
}
.fa-flag:before {
  content: "\f024";
}
.fa-headphones:before {
  content: "\f025";
}
.fa-volume-off:before {
  content: "\f026";
}
.fa-volume-down:before {
  content: "\f027";
}
.fa-volume-up:before {
  content: "\f028";
}
.fa-qrcode:before {
  content: "\f029";
}
.fa-barcode:before {
  content: "\f02a";
}
.fa-tag:before {
  content: "\f02b";
}
.fa-tags:before {
  content: "\f02c";
}
.fa-book:before {
  content: "\f02d";
}
.fa-bookmark:before {
  content: "\f02e";
}
.fa-print:before {
  content: "\f02f";
}
.fa-camera:before {
  content: "\f030";
}
.fa-font:before {
  content: "\f031";
}
.fa-bold:before {
  content: "\f032";
}
.fa-italic:before {
  content: "\f033";
}
.fa-text-height:before {
  content: "\f034";
}
.fa-text-width:before {
  content: "\f035";
}
.fa-align-left:before {
  content: "\f036";
}
.fa-align-center:before {
  content: "\f037";
}
.fa-align-right:before {
  content: "\f038";
}
.fa-align-justify:before {
  content: "\f039";
}
.fa-list:before {
  content: "\f03a";
}
.fa-dedent:before,
.fa-outdent:before {
  content: "\f03b";
}
.fa-indent:before {
  content: "\f03c";
}
.fa-video-camera:before {
  content: "\f03d";
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: "\f03e";
}
.fa-pencil:before {
  content: "\f040";
}
.fa-map-marker:before {
  content: "\f041";
}
.fa-adjust:before {
  content: "\f042";
}
.fa-tint:before {
  content: "\f043";
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: "\f044";
}
.fa-share-square-o:before {
  content: "\f045";
}
.fa-check-square-o:before {
  content: "\f046";
}
.fa-arrows:before {
  content: "\f047";
}
.fa-step-backward:before {
  content: "\f048";
}
.fa-fast-backward:before {
  content: "\f049";
}
.fa-backward:before {
  content: "\f04a";
}
.fa-play:before {
  content: "\f04b";
}
.fa-pause:before {
  content: "\f04c";
}
.fa-stop:before {
  content: "\f04d";
}
.fa-forward:before {
  content: "\f04e";
}
.fa-fast-forward:before {
  content: "\f050";
}
.fa-step-forward:before {
  content: "\f051";
}
.fa-eject:before {
  content: "\f052";
}
.fa-chevron-left:before {
  content: "\f053";
}
.fa-chevron-right:before {
  content: "\f054";
}
.fa-plus-circle:before {
  content: "\f055";
}
.fa-minus-circle:before {
  content: "\f056";
}
.fa-times-circle:before {
  content: "\f057";
}
.fa-check-circle:before {
  content: "\f058";
}
.fa-question-circle:before {
  content: "\f059";
}
.fa-info-circle:before {
  content: "\f05a";
}
.fa-crosshairs:before {
  content: "\f05b";
}
.fa-times-circle-o:before {
  content: "\f05c";
}
.fa-check-circle-o:before {
  content: "\f05d";
}
.fa-ban:before {
  content: "\f05e";
}
.fa-arrow-left:before {
  content: "\f060";
}
.fa-arrow-right:before {
  content: "\f061";
}
.fa-arrow-up:before {
  content: "\f062";
}
.fa-arrow-down:before {
  content: "\f063";
}
.fa-mail-forward:before,
.fa-share:before {
  content: "\f064";
}
.fa-expand:before {
  content: "\f065";
}
.fa-compress:before {
  content: "\f066";
}
.fa-plus:before {
  content: "\f067";
}
.fa-minus:before {
  content: "\f068";
}
.fa-asterisk:before {
  content: "\f069";
}
.fa-exclamation-circle:before {
  content: "\f06a";
}
.fa-gift:before {
  content: "\f06b";
}
.fa-leaf:before {
  content: "\f06c";
}
.fa-fire:before {
  content: "\f06d";
}
.fa-eye:before {
  content: "\f06e";
}
.fa-eye-slash:before {
  content: "\f070";
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: "\f071";
}
.fa-plane:before {
  content: "\f072";
}
.fa-calendar:before {
  content: "\f073";
}
.fa-random:before {
  content: "\f074";
}
.fa-comment:before {
  content: "\f075";
}
.fa-magnet:before {
  content: "\f076";
}
.fa-chevron-up:before {
  content: "\f077";
}
.fa-chevron-down:before {
  content: "\f078";
}
.fa-retweet:before {
  content: "\f079";
}
.fa-shopping-cart:before {
  content: "\f07a";
}
.fa-folder:before {
  content: "\f07b";
}
.fa-folder-open:before {
  content: "\f07c";
}
.fa-arrows-v:before {
  content: "\f07d";
}
.fa-arrows-h:before {
  content: "\f07e";
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: "\f080";
}
.fa-twitter-square:before {
  content: "\f081";
}
.fa-facebook-square:before {
  content: "\f082";
}
.fa-camera-retro:before {
  content: "\f083";
}
.fa-key:before {
  content: "\f084";
}
.fa-gears:before,
.fa-cogs:before {
  content: "\f085";
}
.fa-comments:before {
  content: "\f086";
}
.fa-thumbs-o-up:before {
  content: "\f087";
}
.fa-thumbs-o-down:before {
  content: "\f088";
}
.fa-star-half:before {
  content: "\f089";
}
.fa-heart-o:before {
  content: "\f08a";
}
.fa-sign-out:before {
  content: "\f08b";
}
.fa-linkedin-square:before {
  content: "\f08c";
}
.fa-thumb-tack:before {
  content: "\f08d";
}
.fa-external-link:before {
  content: "\f08e";
}
.fa-sign-in:before {
  content: "\f090";
}
.fa-trophy:before {
  content: "\f091";
}
.fa-github-square:before {
  content: "\f092";
}
.fa-upload:before {
  content: "\f093";
}
.fa-lemon-o:before {
  content: "\f094";
}
.fa-phone:before {
  content: "\f095";
}
.fa-square-o:before {
  content: "\f096";
}
.fa-bookmark-o:before {
  content: "\f097";
}
.fa-phone-square:before {
  content: "\f098";
}
.fa-twitter:before {
  content: "\f099";
}
.fa-facebook:before {
  content: "\f09a";
}
.fa-github:before {
  content: "\f09b";
}
.fa-unlock:before {
  content: "\f09c";
}
.fa-credit-card:before {
  content: "\f09d";
}
.fa-rss:before {
  content: "\f09e";
}
.fa-hdd-o:before {
  content: "\f0a0";
}
.fa-bullhorn:before {
  content: "\f0a1";
}
.fa-bell:before {
  content: "\f0f3";
}
.fa-certificate:before {
  content: "\f0a3";
}
.fa-hand-o-right:before {
  content: "\f0a4";
}
.fa-hand-o-left:before {
  content: "\f0a5";
}
.fa-hand-o-up:before {
  content: "\f0a6";
}
.fa-hand-o-down:before {
  content: "\f0a7";
}
.fa-arrow-circle-left:before {
  content: "\f0a8";
}
.fa-arrow-circle-right:before {
  content: "\f0a9";
}
.fa-arrow-circle-up:before {
  content: "\f0aa";
}
.fa-arrow-circle-down:before {
  content: "\f0ab";
}
.fa-globe:before {
  content: "\f0ac";
}
.fa-wrench:before {
  content: "\f0ad";
}
.fa-tasks:before {
  content: "\f0ae";
}
.fa-filter:before {
  content: "\f0b0";
}
.fa-briefcase:before {
  content: "\f0b1";
}
.fa-arrows-alt:before {
  content: "\f0b2";
}
.fa-group:before,
.fa-users:before {
  content: "\f0c0";
}
.fa-chain:before,
.fa-link:before {
  content: "\f0c1";
}
.fa-cloud:before {
  content: "\f0c2";
}
.fa-flask:before {
  content: "\f0c3";
}
.fa-cut:before,
.fa-scissors:before {
  content: "\f0c4";
}
.fa-copy:before,
.fa-files-o:before {
  content: "\f0c5";
}
.fa-paperclip:before {
  content: "\f0c6";
}
.fa-save:before,
.fa-floppy-o:before {
  content: "\f0c7";
}
.fa-square:before {
  content: "\f0c8";
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: "\f0c9";
}
.fa-list-ul:before {
  content: "\f0ca";
}
.fa-list-ol:before {
  content: "\f0cb";
}
.fa-strikethrough:before {
  content: "\f0cc";
}
.fa-underline:before {
  content: "\f0cd";
}
.fa-table:before {
  content: "\f0ce";
}
.fa-magic:before {
  content: "\f0d0";
}
.fa-truck:before {
  content: "\f0d1";
}
.fa-pinterest:before {
  content: "\f0d2";
}
.fa-pinterest-square:before {
  content: "\f0d3";
}
.fa-google-plus-square:before {
  content: "\f0d4";
}
.fa-google-plus:before {
  content: "\f0d5";
}
.fa-money:before {
  content: "\f0d6";
}
.fa-caret-down:before {
  content: "\f0d7";
}
.fa-caret-up:before {
  content: "\f0d8";
}
.fa-caret-left:before {
  content: "\f0d9";
}
.fa-caret-right:before {
  content: "\f0da";
}
.fa-columns:before {
  content: "\f0db";
}
.fa-unsorted:before,
.fa-sort:before {
  content: "\f0dc";
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: "\f0dd";
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: "\f0de";
}
.fa-envelope:before {
  content: "\f0e0";
}
.fa-linkedin:before {
  content: "\f0e1";
}
.fa-rotate-left:before,
.fa-undo:before {
  content: "\f0e2";
}
.fa-legal:before,
.fa-gavel:before {
  content: "\f0e3";
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: "\f0e4";
}
.fa-comment-o:before {
  content: "\f0e5";
}
.fa-comments-o:before {
  content: "\f0e6";
}
.fa-flash:before,
.fa-bolt:before {
  content: "\f0e7";
}
.fa-sitemap:before {
  content: "\f0e8";
}
.fa-umbrella:before {
  content: "\f0e9";
}
.fa-paste:before,
.fa-clipboard:before {
  content: "\f0ea";
}
.fa-lightbulb-o:before {
  content: "\f0eb";
}
.fa-exchange:before {
  content: "\f0ec";
}
.fa-cloud-download:before {
  content: "\f0ed";
}
.fa-cloud-upload:before {
  content: "\f0ee";
}
.fa-user-md:before {
  content: "\f0f0";
}
.fa-stethoscope:before {
  content: "\f0f1";
}
.fa-suitcase:before {
  content: "\f0f2";
}
.fa-bell-o:before {
  content: "\f0a2";
}
.fa-coffee:before {
  content: "\f0f4";
}
.fa-cutlery:before {
  content: "\f0f5";
}
.fa-file-text-o:before {
  content: "\f0f6";
}
.fa-building-o:before {
  content: "\f0f7";
}
.fa-hospital-o:before {
  content: "\f0f8";
}
.fa-ambulance:before {
  content: "\f0f9";
}
.fa-medkit:before {
  content: "\f0fa";
}
.fa-fighter-jet:before {
  content: "\f0fb";
}
.fa-beer:before {
  content: "\f0fc";
}
.fa-h-square:before {
  content: "\f0fd";
}
.fa-plus-square:before {
  content: "\f0fe";
}
.fa-angle-double-left:before {
  content: "\f100";
}
.fa-angle-double-right:before {
  content: "\f101";
}
.fa-angle-double-up:before {
  content: "\f102";
}
.fa-angle-double-down:before {
  content: "\f103";
}
.fa-angle-left:before {
  content: "\f104";
}
.fa-angle-right:before {
  content: "\f105";
}
.fa-angle-up:before {
  content: "\f106";
}
.fa-angle-down:before {
  content: "\f107";
}
.fa-desktop:before {
  content: "\f108";
}
.fa-laptop:before {
  content: "\f109";
}
.fa-tablet:before {
  content: "\f10a";
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: "\f10b";
}
.fa-circle-o:before {
  content: "\f10c";
}
.fa-quote-left:before {
  content: "\f10d";
}
.fa-quote-right:before {
  content: "\f10e";
}
.fa-spinner:before {
  content: "\f110";
}
.fa-circle:before {
  content: "\f111";
}
.fa-mail-reply:before,
.fa-reply:before {
  content: "\f112";
}
.fa-github-alt:before {
  content: "\f113";
}
.fa-folder-o:before {
  content: "\f114";
}
.fa-folder-open-o:before {
  content: "\f115";
}
.fa-smile-o:before {
  content: "\f118";
}
.fa-frown-o:before {
  content: "\f119";
}
.fa-meh-o:before {
  content: "\f11a";
}
.fa-gamepad:before {
  content: "\f11b";
}
.fa-keyboard-o:before {
  content: "\f11c";
}
.fa-flag-o:before {
  content: "\f11d";
}
.fa-flag-checkered:before {
  content: "\f11e";
}
.fa-terminal:before {
  content: "\f120";
}
.fa-code:before {
  content: "\f121";
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: "\f122";
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: "\f123";
}
.fa-location-arrow:before {
  content: "\f124";
}
.fa-crop:before {
  content: "\f125";
}
.fa-code-fork:before {
  content: "\f126";
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: "\f127";
}
.fa-question:before {
  content: "\f128";
}
.fa-info:before {
  content: "\f129";
}
.fa-exclamation:before {
  content: "\f12a";
}
.fa-superscript:before {
  content: "\f12b";
}
.fa-subscript:before {
  content: "\f12c";
}
.fa-eraser:before {
  content: "\f12d";
}
.fa-puzzle-piece:before {
  content: "\f12e";
}
.fa-microphone:before {
  content: "\f130";
}
.fa-microphone-slash:before {
  content: "\f131";
}
.fa-shield:before {
  content: "\f132";
}
.fa-calendar-o:before {
  content: "\f133";
}
.fa-fire-extinguisher:before {
  content: "\f134";
}
.fa-rocket:before {
  content: "\f135";
}
.fa-maxcdn:before {
  content: "\f136";
}
.fa-chevron-circle-left:before {
  content: "\f137";
}
.fa-chevron-circle-right:before {
  content: "\f138";
}
.fa-chevron-circle-up:before {
  content: "\f139";
}
.fa-chevron-circle-down:before {
  content: "\f13a";
}
.fa-html5:before {
  content: "\f13b";
}
.fa-css3:before {
  content: "\f13c";
}
.fa-anchor:before {
  content: "\f13d";
}
.fa-unlock-alt:before {
  content: "\f13e";
}
.fa-bullseye:before {
  content: "\f140";
}
.fa-ellipsis-h:before {
  content: "\f141";
}
.fa-ellipsis-v:before {
  content: "\f142";
}
.fa-rss-square:before {
  content: "\f143";
}
.fa-play-circle:before {
  content: "\f144";
}
.fa-ticket:before {
  content: "\f145";
}
.fa-minus-square:before {
  content: "\f146";
}
.fa-minus-square-o:before {
  content: "\f147";
}
.fa-level-up:before {
  content: "\f148";
}
.fa-level-down:before {
  content: "\f149";
}
.fa-check-square:before {
  content: "\f14a";
}
.fa-pencil-square:before {
  content: "\f14b";
}
.fa-external-link-square:before {
  content: "\f14c";
}
.fa-share-square:before {
  content: "\f14d";
}
.fa-compass:before {
  content: "\f14e";
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: "\f150";
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: "\f151";
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: "\f152";
}
.fa-euro:before,
.fa-eur:before {
  content: "\f153";
}
.fa-gbp:before {
  content: "\f154";
}
.fa-dollar:before,
.fa-usd:before {
  content: "\f155";
}
.fa-rupee:before,
.fa-inr:before {
  content: "\f156";
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: "\f157";
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: "\f158";
}
.fa-won:before,
.fa-krw:before {
  content: "\f159";
}
.fa-bitcoin:before,
.fa-btc:before {
  content: "\f15a";
}
.fa-file:before {
  content: "\f15b";
}
.fa-file-text:before {
  content: "\f15c";
}
.fa-sort-alpha-asc:before {
  content: "\f15d";
}
.fa-sort-alpha-desc:before {
  content: "\f15e";
}
.fa-sort-amount-asc:before {
  content: "\f160";
}
.fa-sort-amount-desc:before {
  content: "\f161";
}
.fa-sort-numeric-asc:before {
  content: "\f162";
}
.fa-sort-numeric-desc:before {
  content: "\f163";
}
.fa-thumbs-up:before {
  content: "\f164";
}
.fa-thumbs-down:before {
  content: "\f165";
}
.fa-youtube-square:before {
  content: "\f166";
}
.fa-youtube:before {
  content: "\f167";
}
.fa-xing:before {
  content: "\f168";
}
.fa-xing-square:before {
  content: "\f169";
}
.fa-youtube-play:before {
  content: "\f16a";
}
.fa-dropbox:before {
  content: "\f16b";
}
.fa-stack-overflow:before {
  content: "\f16c";
}
.fa-instagram:before {
  content: "\f16d";
}
.fa-flickr:before {
  content: "\f16e";
}
.fa-adn:before {
  content: "\f170";
}
.fa-bitbucket:before {
  content: "\f171";
}
.fa-bitbucket-square:before {
  content: "\f172";
}
.fa-tumblr:before {
  content: "\f173";
}
.fa-tumblr-square:before {
  content: "\f174";
}
.fa-long-arrow-down:before {
  content: "\f175";
}
.fa-long-arrow-up:before {
  content: "\f176";
}
.fa-long-arrow-left:before {
  content: "\f177";
}
.fa-long-arrow-right:before {
  content: "\f178";
}
.fa-apple:before {
  content: "\f179";
}
.fa-windows:before {
  content: "\f17a";
}
.fa-android:before {
  content: "\f17b";
}
.fa-linux:before {
  content: "\f17c";
}
.fa-dribbble:before {
  content: "\f17d";
}
.fa-skype:before {
  content: "\f17e";
}
.fa-foursquare:before {
  content: "\f180";
}
.fa-trello:before {
  content: "\f181";
}
.fa-female:before {
  content: "\f182";
}
.fa-male:before {
  content: "\f183";
}
.fa-gittip:before {
  content: "\f184";
}
.fa-sun-o:before {
  content: "\f185";
}
.fa-moon-o:before {
  content: "\f186";
}
.fa-archive:before {
  content: "\f187";
}
.fa-bug:before {
  content: "\f188";
}
.fa-vk:before {
  content: "\f189";
}
.fa-weibo:before {
  content: "\f18a";
}
.fa-renren:before {
  content: "\f18b";
}
.fa-pagelines:before {
  content: "\f18c";
}
.fa-stack-exchange:before {
  content: "\f18d";
}
.fa-arrow-circle-o-right:before {
  content: "\f18e";
}
.fa-arrow-circle-o-left:before {
  content: "\f190";
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: "\f191";
}
.fa-dot-circle-o:before {
  content: "\f192";
}
.fa-wheelchair:before {
  content: "\f193";
}
.fa-vimeo-square:before {
  content: "\f194";
}
.fa-turkish-lira:before,
.fa-try:before {
  content: "\f195";
}
.fa-plus-square-o:before {
  content: "\f196";
}
.fa-space-shuttle:before {
  content: "\f197";
}
.fa-slack:before {
  content: "\f198";
}
.fa-envelope-square:before {
  content: "\f199";
}
.fa-wordpress:before {
  content: "\f19a";
}
.fa-openid:before {
  content: "\f19b";
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: "\f19c";
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: "\f19d";
}
.fa-yahoo:before {
  content: "\f19e";
}
.fa-google:before {
  content: "\f1a0";
}
.fa-reddit:before {
  content: "\f1a1";
}
.fa-reddit-square:before {
  content: "\f1a2";
}
.fa-stumbleupon-circle:before {
  content: "\f1a3";
}
.fa-stumbleupon:before {
  content: "\f1a4";
}
.fa-delicious:before {
  content: "\f1a5";
}
.fa-digg:before {
  content: "\f1a6";
}
.fa-pied-piper:before {
  content: "\f1a7";
}
.fa-pied-piper-alt:before {
  content: "\f1a8";
}
.fa-drupal:before {
  content: "\f1a9";
}
.fa-joomla:before {
  content: "\f1aa";
}
.fa-language:before {
  content: "\f1ab";
}
.fa-fax:before {
  content: "\f1ac";
}
.fa-building:before {
  content: "\f1ad";
}
.fa-child:before {
  content: "\f1ae";
}
.fa-paw:before {
  content: "\f1b0";
}
.fa-spoon:before {
  content: "\f1b1";
}
.fa-cube:before {
  content: "\f1b2";
}
.fa-cubes:before {
  content: "\f1b3";
}
.fa-behance:before {
  content: "\f1b4";
}
.fa-behance-square:before {
  content: "\f1b5";
}
.fa-steam:before {
  content: "\f1b6";
}
.fa-steam-square:before {
  content: "\f1b7";
}
.fa-recycle:before {
  content: "\f1b8";
}
.fa-automobile:before,
.fa-car:before {
  content: "\f1b9";
}
.fa-cab:before,
.fa-taxi:before {
  content: "\f1ba";
}
.fa-tree:before {
  content: "\f1bb";
}
.fa-spotify:before {
  content: "\f1bc";
}
.fa-deviantart:before {
  content: "\f1bd";
}
.fa-soundcloud:before {
  content: "\f1be";
}
.fa-database:before {
  content: "\f1c0";
}
.fa-file-pdf-o:before {
  content: "\f1c1";
}
.fa-file-word-o:before {
  content: "\f1c2";
}
.fa-file-excel-o:before {
  content: "\f1c3";
}
.fa-file-powerpoint-o:before {
  content: "\f1c4";
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: "\f1c5";
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: "\f1c6";
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: "\f1c7";
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: "\f1c8";
}
.fa-file-code-o:before {
  content: "\f1c9";
}
.fa-vine:before {
  content: "\f1ca";
}
.fa-codepen:before {
  content: "\f1cb";
}
.fa-jsfiddle:before {
  content: "\f1cc";
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: "\f1cd";
}
.fa-circle-o-notch:before {
  content: "\f1ce";
}
.fa-ra:before,
.fa-rebel:before {
  content: "\f1d0";
}
.fa-ge:before,
.fa-empire:before {
  content: "\f1d1";
}
.fa-git-square:before {
  content: "\f1d2";
}
.fa-git:before {
  content: "\f1d3";
}
.fa-hacker-news:before {
  content: "\f1d4";
}
.fa-tencent-weibo:before {
  content: "\f1d5";
}
.fa-qq:before {
  content: "\f1d6";
}
.fa-wechat:before,
.fa-weixin:before {
  content: "\f1d7";
}
.fa-send:before,
.fa-paper-plane:before {
  content: "\f1d8";
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: "\f1d9";
}
.fa-history:before {
  content: "\f1da";
}
.fa-circle-thin:before {
  content: "\f1db";
}
.fa-header:before {
  content: "\f1dc";
}
.fa-paragraph:before {
  content: "\f1dd";
}
.fa-sliders:before {
  content: "\f1de";
}
.fa-share-alt:before {
  content: "\f1e0";
}
.fa-share-alt-square:before {
  content: "\f1e1";
}
.fa-bomb:before {
  content: "\f1e2";
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: "\f1e3";
}
.fa-tty:before {
  content: "\f1e4";
}
.fa-binoculars:before {
  content: "\f1e5";
}
.fa-plug:before {
  content: "\f1e6";
}
.fa-slideshare:before {
  content: "\f1e7";
}
.fa-twitch:before {
  content: "\f1e8";
}
.fa-yelp:before {
  content: "\f1e9";
}
.fa-newspaper-o:before {
  content: "\f1ea";
}
.fa-wifi:before {
  content: "\f1eb";
}
.fa-calculator:before {
  content: "\f1ec";
}
.fa-paypal:before {
  content: "\f1ed";
}
.fa-google-wallet:before {
  content: "\f1ee";
}
.fa-cc-visa:before {
  content: "\f1f0";
}
.fa-cc-mastercard:before {
  content: "\f1f1";
}
.fa-cc-discover:before {
  content: "\f1f2";
}
.fa-cc-amex:before {
  content: "\f1f3";
}
.fa-cc-paypal:before {
  content: "\f1f4";
}
.fa-cc-stripe:before {
  content: "\f1f5";
}
.fa-bell-slash:before {
  content: "\f1f6";
}
.fa-bell-slash-o:before {
  content: "\f1f7";
}
.fa-trash:before {
  content: "\f1f8";
}
.fa-copyright:before {
  content: "\f1f9";
}
.fa-at:before {
  content: "\f1fa";
}
.fa-eyedropper:before {
  content: "\f1fb";
}
.fa-paint-brush:before {
  content: "\f1fc";
}
.fa-birthday-cake:before {
  content: "\f1fd";
}
.fa-area-chart:before {
  content: "\f1fe";
}
.fa-pie-chart:before {
  content: "\f200";
}
.fa-line-chart:before {
  content: "\f201";
}
.fa-lastfm:before {
  content: "\f202";
}
.fa-lastfm-square:before {
  content: "\f203";
}
.fa-toggle-off:before {
  content: "\f204";
}
.fa-toggle-on:before {
  content: "\f205";
}
.fa-bicycle:before {
  content: "\f206";
}
.fa-bus:before {
  content: "\f207";
}
.fa-ioxhost:before {
  content: "\f208";
}
.fa-angellist:before {
  content: "\f209";
}
.fa-cc:before {
  content: "\f20a";
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: "\f20b";
}
.fa-meanpath:before {
  content: "\f20c";
}
/*!
*
* IPython base
*
*/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/* Make the page background atleast 100% the height of the view port */
/* Make the page itself atleast 70% the height of the view port */
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/* Flexible box model classes */
/* Taken from Alex Russell http://infrequently.org/2009/08/css-3-progress/ */
/* This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
`Modern browsers` comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
`Old browsers` comments below.
*/
.hbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.vbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /* Old browsers */
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /* Modern browsers */
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /* Old browsers */
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /* Old browsers */
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /* Modern browsers */
  flex: 2;
}
.box-group1 {
  /*  Deprecated */
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /* Deprecated */
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /* Old browsers */
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /* Modern browsers */
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /* Old browsers */
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /* Modern browsers */
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /* Old browsers */
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /* Modern browsers */
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /* Old browsers */
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /* Modern browsers */
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /* Old browsers */
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /* Modern browsers */
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /* Old browsers */
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /* Modern browsers */
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /* Old browsers */
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /* Modern browsers */
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /* Old browsers */
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /* Modern browsers */
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /* Old browsers */
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /* Modern browsers */
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error > h1 {
  font-size: 500%;
  line-height: normal;
}
div.error > p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
body {
  background-color: #fff;
  /* This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below */
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body > #header {
  /* Initially hidden to prevent FLOUC */
  display: none;
  background-color: #fff;
  /* Display over codemirror */
  position: relative;
  z-index: 100;
}
body > #header #header-container {
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body > #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body > #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
@media (max-width: 991px) {
  #ipython_notebook {
    margin-left: 10px;
  }
}
[dir="rtl"] #ipython_notebook {
  float: right !important;
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/* Smaller buttons */
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#login_widget {
  float: right;
}
span#login_widget > .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button:focus,
#logout:focus,
span#login_widget > .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget > .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active:hover,
#logout:active:hover,
span#login_widget > .button.active:hover,
#logout.active:hover,
.open > .dropdown-togglespan#login_widget > .button:hover,
.open > .dropdown-toggle#logout:hover,
span#login_widget > .button:active:focus,
#logout:active:focus,
span#login_widget > .button.active:focus,
#logout.active:focus,
.open > .dropdown-togglespan#login_widget > .button:focus,
.open > .dropdown-toggle#logout:focus,
span#login_widget > .button:active.focus,
#logout:active.focus,
span#login_widget > .button.active.focus,
#logout.active.focus,
.open > .dropdown-togglespan#login_widget > .button.focus,
.open > .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget > .button.disabled:hover,
#logout.disabled:hover,
span#login_widget > .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget > .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget > .button.disabled:focus,
#logout.disabled:focus,
span#login_widget > .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget > .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget > .button.disabled.focus,
#logout.disabled.focus,
span#login_widget > .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget > .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header > span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/*!
*
* IPython auth
*
*/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
/*!
*
* IPython tree view
*
*/
/* We need an invisible input field on top of the sentense*/
/* "Drag file onto the list ..." */
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  text-align: center;
  vertical-align: middle;
  display: inline;
  opacity: 0;
  z-index: 2;
  width: 12ex;
  margin-right: -12ex;
}
.alternate_upload .btn-upload {
  height: 22px;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
[dir="rtl"] #tabs li {
  float: right;
}
ul#tabs {
  margin-bottom: 4px;
}
[dir="rtl"] ul#tabs {
  margin-right: 0px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
[dir="rtl"] .list_toolbar .tree-buttons {
  float: left !important;
}
[dir="rtl"] .list_toolbar .pull-right {
  padding-top: 1px;
  float: left !important;
}
[dir="rtl"] .list_toolbar .pull-left {
  float: right !important;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class*="span"] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container > div {
  border-bottom: 1px solid #ddd;
}
.list_container > div:hover .list-item {
  background-color: red;
}
.list_container > div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header > div,
.list_item > div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header > div input,
.list_item > div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: baseline;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header > div .item_link,
.list_item > div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons > .btn,
.item_buttons > .btn-group,
.item_buttons > .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name > .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
#tree-selector {
  padding-right: 0px;
}
[dir="rtl"] #tree-selector a {
  float: right;
}
#button-select-all {
  min-width: 50px;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f114";
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f016";
  position: relative;
  top: -2px;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
[dir="rtl"] #new-menu {
  text-align: right;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: "\f096";
}
.kernel-menu-icon:before {
  content: "\f096";
}
.kernel-menu-icon-current:before {
  content: "\f00c";
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
[dir="rtl"] #running .col-sm-8 {
  float: right !important;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/*!
*
* IPython text editor webapp
*
*/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: "\f00c";
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /* Use a negative 1 bottom margin, so the border overlaps the border of the
    header */
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f00c";
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI colors. */
.ansibold {
  font-weight: bold;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  border-left-width: 1px;
  padding-left: 5px;
  background: linear-gradient(to right, transparent -40px, transparent 1px, transparent 1px, transparent 100%);
}
div.cell.jupyter-soft-selected {
  border-left-color: #90CAF9;
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected {
  border-color: #ababab;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 5px, transparent 5px, transparent 100%);
}
@media print {
  div.cell.selected {
    border-color: transparent;
  }
}
div.cell.selected.jupyter-soft-selected {
  border-left-width: 0;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 7px, #E3F2FD 7px, #E3F2FD 100%);
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #66BB6A -40px, #66BB6A 5px, transparent 5px, transparent 100%);
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  min-width: 0;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  padding: 0.4em;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. We need the 0 value because of how we size */
  /* .CodeMirror-lines */
  padding: 0;
  border: 0;
  border-radius: 0;
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul {
  list-style: disc;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ul ul {
  list-style: square;
  margin: 0em 2em;
}
.rendered_html ul ul ul {
  list-style: circle;
  margin: 0em 2em;
}
.rendered_html ol {
  list-style: decimal;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
  margin: 0em 2em;
}
.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  background-color: #fff;
  color: #000;
  font-size: 100%;
  padding: 0px;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: 1px solid black;
  border-collapse: collapse;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  border: 1px solid black;
  border-collapse: collapse;
  margin: 1em 2em;
}
.rendered_html td,
.rendered_html th {
  text-align: left;
  vertical-align: middle;
  padding: 4px;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html * + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html * + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/*!
*
* IPython notebook webapp
*
*/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /* This spaces the page away from the edge of the notebook area */
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/* Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 */
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app > #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/* CSS for the cell toolbar */
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/* ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
*/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget {
  float: right !important;
  float: right;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu > .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
.dropdown-submenu:hover > .dropdown-menu {
  display: block;
}
.dropdown-submenu > a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: "\f0da";
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu > a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.pull-right {
  margin-left: .3em;
}
.dropdown-submenu:hover > a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left > .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f040";
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: ' ';
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f10c";
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f111";
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f1e2";
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f127";
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open > .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open > .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open > .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open > .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open > .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open > .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open > .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open > .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open > .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open > .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open > .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open > .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open > .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open > .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open > .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /* Display over codemirror */
  z-index: 100;
  /* Hack which prevents jquery ui resizable from changing top. */
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /* This injects handle bars (a short, wide = symbol) for 
        the resize handle. */
}
div#pager .ui-resizable-handle::after {
  content: '';
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 21ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
span.save_widget {
  margin-top: 6px;
}
span.save_widget span.filename {
  height: 1em;
  line-height: 1em;
  padding: 3px;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
.pulse,
.dropdown-menu > li > a.pulse,
li.pulse > a.dropdown-toggle,
li.pulse.open > a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
/** WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 * of chance of beeing generated from the ../less/[samename].less file, you can
 * try to get back the less file by reverting somme commit in history
 **/
/*
 * We'll try to get something pretty, so we
 * have some strange css to have the scroll bar on
 * the left with fix button on the top right of the tooltip
 */
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/*properties of tooltip after "expand"*/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/*properties of tooltip before "expand"*/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /*avoid the button to overlap on some docstring*/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /*fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: "";
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list > li > a {
  /** Firefox bug **/
  /* see https://github.com/jupyter/notebook/issues/559 */
  white-space: normal;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  display: none;
}
.command-shortcut:before {
  content: "(command)";
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: "(edit)";
  padding-right: 3px;
  color: #777777;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  width: 100%;
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal .xterm-rows {
  padding: 10px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/*# sourceMappingURL=style.min.css.map */
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
<style type="text/css">
    
/* Temporary definitions which will become obsolete with Notebook release 5.0 */
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }

    </style>


<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
    <!-- End of mathjax configuration --></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Variational-AutoEncoder-in-Pytorch">Variational AutoEncoder in Pytorch<a class="anchor-link" href="#Variational-AutoEncoder-in-Pytorch">&#182;</a></h1><p>For today's quick project, we're going to reproduce a classic unsuperviser network in Pytorch.</p>
<h2 id="Recipe-difficulty:">Recipe difficulty:<a class="anchor-link" href="#Recipe-difficulty:">&#182;</a></h2><ul>
<li>Statistics: ?????? - you don't need any specific knowledge, but you need to grasp the concept of latent gaussians. </li>
<li>Technical: ?? - nothing wild here, and in fact the network is so small you don't even need CUDA</li>
<li>Time required: ?? - training iterations are fast, and the code is short and to the point. It took me 83 minutes total to write this.</li>
</ul>
<h2 id="What-it-is">What it is<a class="anchor-link" href="#What-it-is">&#182;</a></h2><p>The statistical intuition between a VAE which is actually quite straightforward: we use an <strong>encoder</strong> network to 'compress' image data down to two (or more) N-dimensional vectors, which represent an array of scale and distribution parameters from a distribution of choice (typically a Gaussian).</p>
<p>A <strong>decoder</strong> network is then used to translate this set of latent distributions back to an image.</p>
<p>The encoder and the decoder are jointly optimized using a simple <strong>mean square error between the ground truth and reconstructed</strong> pixels. This is the first half of the loss function.</p>
<p>The second part of the loss function is the <strong>Kullback-Leibler divergence</strong>, which can be seen as a regularization term between our latent set of distributions and a set of gaussians. This is to make sure that all latent sampled are coming from the same distribution (i.e. we want our set of 20 sampled parameters to be probabilistically close to the 'true' mu and sigma in the sample space.</p>
<p>As usual, this is by no mean an exhaustive explanation - <a href="http://kvfrans.com/variational-autoencoders-explained/">Kevin Frans' amazing blogpost</a> is a good starting point if you want to know more.</p>
<h2 id="Why-it-is-useful">Why it is useful<a class="anchor-link" href="#Why-it-is-useful">&#182;</a></h2><p>A VAE is a fully unsupervised dimensionality reduction method. It can be useful, for instance, to reduce related data down to a more manageable number of variables, without needing any labelling work.</p>
<p>Note that VAEs can also work on non-image data!</p>
<p>Finally, they are the easiest to grasp out of all generative models. In this post I'll show how to sample the latent distribution to generate completely new data</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We start with the usual imports. I'm using Pytorch here, as it allows for a more understandable joint optimization than say, Keras.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">torch</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">torch.utils.data</span> <span class="k">import</span> <span class="n">Dataset</span><span class="p">,</span> <span class="n">DataLoader</span>
<span class="kn">from</span> <span class="nn">torch</span> <span class="k">import</span> <span class="n">nn</span>
<span class="kn">from</span> <span class="nn">torchvision</span> <span class="k">import</span> <span class="n">transforms</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">torchvision</span>
<span class="kn">import</span> <span class="nn">torch.nn.functional</span> <span class="k">as</span> <span class="nn">F</span>
<span class="kn">from</span> <span class="nn">torch.autograd</span> <span class="k">import</span> <span class="n">Variable</span>
<span class="kn">from</span> <span class="nn">torch</span> <span class="k">import</span> <span class="n">optim</span>

<span class="o">%</span><span class="k">matplotlib</span> inline

<span class="n">GPU</span> <span class="o">=</span> <span class="kc">True</span>

<span class="k">if</span> <span class="ow">not</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">is_available</span><span class="p">():</span>
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;no GPU detected, check CUDA install&#39;</span><span class="p">)</span>
    <span class="n">GPU</span> <span class="o">=</span> <span class="kc">False</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We use Fashion MNIST, which is a more interesting drop-in replacement for the classic MNIST digits dataset as our image source.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">trans</span> <span class="o">=</span> <span class="n">transforms</span><span class="o">.</span><span class="n">Compose</span><span class="p">([</span><span class="n">transforms</span><span class="o">.</span><span class="n">ToTensor</span><span class="p">(),</span>
                                    <span class="n">transforms</span><span class="o">.</span><span class="n">Normalize</span><span class="p">((</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">),</span> <span class="p">(</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">))])</span>

<span class="n">dataset</span> <span class="o">=</span> <span class="n">torchvision</span><span class="o">.</span><span class="n">datasets</span><span class="o">.</span><span class="n">FashionMNIST</span><span class="p">(</span><span class="s1">&#39;day1/data&#39;</span><span class="p">,</span> <span class="n">download</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> 
                                            <span class="n">transform</span><span class="o">=</span><span class="n">trans</span><span class="p">)</span>
<span class="n">dataset</span>
<span class="n">dataloader</span> <span class="o">=</span> <span class="n">DataLoader</span><span class="p">(</span><span class="n">dataset</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">128</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">classes</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;T-shirt/top&#39;</span><span class="p">,</span><span class="s1">&#39;Trouser&#39;</span><span class="p">,</span><span class="s1">&#39;Pullover&#39;</span><span class="p">,</span><span class="s1">&#39;Dress&#39;</span><span class="p">,</span><span class="s1">&#39;Coat&#39;</span><span class="p">,</span>
           <span class="s1">&#39;Sandal&#39;</span><span class="p">,</span><span class="s1">&#39;Shirt&#39;</span><span class="p">,</span><span class="s1">&#39;Sneaker&#39;</span><span class="p">,</span><span class="s1">&#39;Bag&#39;</span><span class="p">,</span><span class="s1">&#39;Ankle boot&#39;</span><span class="p">]</span>

<span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">6</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">2</span><span class="p">))</span>
<span class="k">for</span> <span class="n">ax</span> <span class="ow">in</span> <span class="n">axes</span><span class="o">.</span><span class="n">ravel</span><span class="p">():</span>
    <span class="n">r</span><span class="o">=</span> <span class="nb">iter</span><span class="p">(</span><span class="n">dataloader</span><span class="p">)</span><span class="o">.</span><span class="n">next</span><span class="p">()</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">r</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">numpy</span><span class="p">()[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">28</span><span class="p">,</span><span class="mi">28</span><span class="p">),</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;bone_r&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">classes</span><span class="p">[</span><span class="n">r</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">numpy</span><span class="p">()[</span><span class="mi">1</span><span class="p">]])</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">&#39;off&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>VAE is our Variational Autoencoder network.
It uses a simple set of linear activations, with input and output equal to the flattened image (28x28 pixels = 784), and a set of 20 activation parameters for both mu and sigma. The latter is supposed to be a logarithmic variable, since there can be no negative scale parameter.</p>
<ul>
<li>The <strong>encode</strong> function reduces the input image to a set of 20 location (mu) and 20 scale (sigma) parameters</li>
<li>The <strong>parametric</strong> function uses these parameter to generate a vector of 20 samples.</li>
<li>The <strong>decode</strong> function projects the samples back into the image space.</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[40]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">class</span> <span class="nc">VAE</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">()</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        
        <span class="bp">self</span><span class="o">.</span><span class="n">input_layer</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">784</span><span class="p">,</span> <span class="mi">400</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">mu</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">400</span><span class="p">,</span> <span class="mi">20</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">sigma</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">400</span><span class="p">,</span> <span class="mi">20</span><span class="p">)</span>
        
        <span class="bp">self</span><span class="o">.</span><span class="n">upscale_sample</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">20</span><span class="p">,</span> <span class="mi">400</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">output_layer</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">400</span><span class="p">,</span> <span class="mi">784</span><span class="p">)</span>
        
    <span class="k">def</span> <span class="nf">encode</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">input_layer</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">mu</span><span class="p">(</span><span class="n">x</span><span class="p">),</span> <span class="bp">self</span><span class="o">.</span><span class="n">sigma</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    
    <span class="k">def</span> <span class="nf">parametric</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">mu</span><span class="p">,</span> <span class="n">sigma</span><span class="p">):</span>
        <span class="c1"># A gaussian with</span>
        <span class="c1"># mu mean between 0 and inf</span>
        <span class="c1"># sigma std between 0 and inf</span>
        
        <span class="n">std</span> <span class="o">=</span> <span class="n">sigma</span><span class="o">.</span><span class="n">exp</span><span class="p">()</span>
        <span class="k">if</span> <span class="n">GPU</span><span class="p">:</span>
            <span class="n">sample</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">FloatTensor</span><span class="p">(</span><span class="n">mu</span><span class="o">.</span><span class="n">size</span><span class="p">())</span><span class="o">.</span><span class="n">normal_</span><span class="p">()</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">sample</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">FloatTensor</span><span class="p">(</span><span class="n">mu</span><span class="o">.</span><span class="n">size</span><span class="p">())</span><span class="o">.</span><span class="n">normal_</span><span class="p">()</span>
            
        <span class="n">sample</span> <span class="o">=</span> <span class="n">Variable</span><span class="p">(</span><span class="n">sample</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">sample</span><span class="o">.</span><span class="n">mul</span><span class="p">(</span><span class="n">std</span><span class="p">)</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">mu</span><span class="p">)</span>
    
    <span class="k">def</span> <span class="nf">decode</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">sample</span><span class="p">):</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">F</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">upscale_sample</span><span class="p">(</span><span class="n">sample</span><span class="p">))</span>
        <span class="k">return</span> <span class="n">torch</span><span class="o">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">output_layer</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
    
    
    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="n">mu</span><span class="p">,</span> <span class="n">sigma</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">encode</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">sample</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">parametric</span><span class="p">(</span><span class="n">mu</span><span class="p">,</span> <span class="n">sigma</span><span class="p">)</span>
        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">sample</span><span class="p">),</span> <span class="n">mu</span><span class="p">,</span> <span class="n">sigma</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">VAE</span><span class="p">()</span>

<span class="k">if</span> <span class="n">GPU</span><span class="p">:</span>
    <span class="n">model</span><span class="o">.</span><span class="n">cuda</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The loss function is, as mentioned, the sum of a mean squared error and the KL divergence.</p>
<p>For the standard normal case, the KL divergence reduces to the below equation. I have derived this back when I studied statistics, but nowadays I just trust the result :)</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[42]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">loss_function</span><span class="p">(</span><span class="n">reconstructed</span><span class="p">,</span> <span class="n">ground_truth</span><span class="p">,</span> <span class="n">mu</span><span class="p">,</span> <span class="n">sigma</span><span class="p">):</span>
    <span class="n">image_loss</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">MSELoss</span><span class="p">(</span><span class="n">reduction</span><span class="o">=</span><span class="s1">&#39;sum&#39;</span><span class="p">)(</span><span class="n">reconstructed</span><span class="p">,</span> <span class="n">ground_truth</span><span class="p">)</span>
    <span class="n">KL_divergence_loss</span> <span class="o">=</span> <span class="o">-</span><span class="mf">0.5</span> <span class="o">*</span> <span class="n">torch</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="mi">1</span> <span class="o">+</span> <span class="n">sigma</span> <span class="o">-</span> <span class="n">mu</span><span class="o">.</span><span class="n">pow</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span> <span class="o">-</span> <span class="n">sigma</span><span class="o">.</span><span class="n">exp</span><span class="p">())</span>
    <span class="k">return</span> <span class="n">image_loss</span><span class="p">,</span> <span class="n">KL_divergence_loss</span>


<span class="n">optimizer</span> <span class="o">=</span> <span class="n">optim</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span><span class="o">=</span><span class="mf">1e-3</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The model stops improving after around 20 epochs.</p>
<p>The key steps here are, as typical for Pytorch (but more involved from people coming from Sklearn or Keras), that you:</p>
<ul>
<li>zero-out the gradient from your optimizer </li>
<li>you calculate the loss function based on input and output from your model</li>
<li>you do a .backward() step on the loss for backprop</li>
<li>you move on to the next step</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[45]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">epochs</span> <span class="o">=</span> <span class="mi">10</span>

<span class="k">for</span> <span class="n">epoch</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">epochs</span><span class="p">):</span>
    <span class="n">model</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
    <span class="n">train_loss_im</span><span class="p">,</span> <span class="n">train_loss_kl</span> <span class="o">=</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span>
    <span class="k">for</span> <span class="n">batch_id</span><span class="p">,</span> <span class="n">data</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">dataloader</span><span class="p">):</span>
        <span class="n">img</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">data</span>
        <span class="c1"># reshape</span>
        <span class="n">img</span> <span class="o">=</span> <span class="n">img</span><span class="o">.</span><span class="n">view</span><span class="p">(</span><span class="n">img</span><span class="o">.</span><span class="n">size</span><span class="p">(</span><span class="mi">0</span><span class="p">),</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
        <span class="n">img</span> <span class="o">=</span> <span class="n">Variable</span><span class="p">(</span><span class="n">img</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">GPU</span><span class="p">:</span>
            <span class="n">img</span> <span class="o">=</span> <span class="n">img</span><span class="o">.</span><span class="n">cuda</span><span class="p">()</span>
        
        <span class="n">optimizer</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>
        <span class="n">batch</span><span class="p">,</span> <span class="n">mu</span><span class="p">,</span> <span class="n">sigma</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">img</span><span class="p">)</span>
        <span class="n">loss_im</span><span class="p">,</span> <span class="n">loss_kl</span> <span class="o">=</span> <span class="n">loss_function</span><span class="p">(</span><span class="n">batch</span><span class="p">,</span> <span class="n">img</span><span class="p">,</span> <span class="n">mu</span><span class="p">,</span> <span class="n">sigma</span><span class="p">)</span>
        <span class="n">loss</span> <span class="o">=</span> <span class="n">loss_im</span> <span class="o">+</span> <span class="n">loss_kl</span>
        <span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
        <span class="n">optimizer</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
        
        <span class="n">train_loss_im</span> <span class="o">+=</span> <span class="n">loss_im</span>
        <span class="n">train_loss_kl</span> <span class="o">+=</span> <span class="n">loss_kl</span>
       
    <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;====&gt; Epoch: </span><span class="si">{}</span><span class="s1"> Average KL loss: </span><span class="si">{:.4f}</span><span class="s1">, Average MSE loss: </span><span class="si">{:.4f}</span><span class="s1">, Average loss: </span><span class="si">{:.4f}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span>
    <span class="n">epoch</span><span class="p">,</span> <span class="n">train_loss_im</span> <span class="o">/</span> <span class="nb">len</span><span class="p">(</span><span class="n">dataloader</span><span class="o">.</span><span class="n">dataset</span><span class="p">),</span> 
        <span class="n">train_loss_kl</span> <span class="o">/</span> <span class="nb">len</span><span class="p">(</span><span class="n">dataloader</span><span class="o">.</span><span class="n">dataset</span><span class="p">),</span> <span class="p">(</span><span class="n">train_loss_im</span><span class="o">+</span><span class="n">train_loss_kl</span><span class="p">)</span><span class="o">/</span> <span class="nb">len</span><span class="p">(</span><span class="n">dataloader</span><span class="o">.</span><span class="n">dataset</span><span class="p">)))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>====&gt; Epoch: 0 Average KL loss: 472.3373, Average MSE loss: 6.8929, Average loss: 479.2302
====&gt; Epoch: 1 Average KL loss: 471.4956, Average MSE loss: 6.8626, Average loss: 478.3582
====&gt; Epoch: 2 Average KL loss: 470.7567, Average MSE loss: 6.8633, Average loss: 477.6199
====&gt; Epoch: 3 Average KL loss: 470.1885, Average MSE loss: 6.8936, Average loss: 477.0821
====&gt; Epoch: 4 Average KL loss: 469.6948, Average MSE loss: 6.8892, Average loss: 476.5840
====&gt; Epoch: 5 Average KL loss: 469.2154, Average MSE loss: 6.9267, Average loss: 476.1422
====&gt; Epoch: 6 Average KL loss: 468.8340, Average MSE loss: 6.9498, Average loss: 475.7838
====&gt; Epoch: 7 Average KL loss: 468.5081, Average MSE loss: 6.9733, Average loss: 475.4815
====&gt; Epoch: 8 Average KL loss: 468.2431, Average MSE loss: 6.9910, Average loss: 475.2341
====&gt; Epoch: 9 Average KL loss: 467.9756, Average MSE loss: 7.0138, Average loss: 474.9893
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The model reconstructs the original inputs decently:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[78]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">eval</span><span class="p">()</span>
<span class="n">ev_img</span> <span class="o">=</span> <span class="nb">iter</span><span class="p">(</span><span class="n">dataloader</span><span class="p">)</span><span class="o">.</span><span class="n">next</span><span class="p">()[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">numpy</span><span class="p">()[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">28</span><span class="p">,</span><span class="mi">28</span><span class="p">)</span>
<span class="n">ev_img_r</span> <span class="o">=</span> <span class="n">ev_img</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>

<span class="n">ev_mu</span><span class="p">,</span> <span class="n">ev_sigma</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">encode</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">from_numpy</span><span class="p">(</span><span class="n">ev_img_r</span><span class="p">)</span><span class="o">.</span><span class="n">cuda</span><span class="p">())</span>
<span class="n">ev_param_tensor</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">parametric</span><span class="p">(</span><span class="n">ev_mu</span><span class="p">,</span><span class="n">ev_sigma</span><span class="p">)</span>
<span class="n">ev_out</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">ev_param_tensor</span><span class="p">)</span><span class="o">.</span><span class="n">cpu</span><span class="p">()</span><span class="o">.</span><span class="n">detach</span><span class="p">()</span><span class="o">.</span><span class="n">numpy</span><span class="p">()</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">28</span><span class="p">,</span><span class="mi">28</span><span class="p">)</span>

<span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;original image&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">ev_img</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;bone_r&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;reconstructed&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">ev_out</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;bone_r&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[78]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>[]</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As a bonus, we can just sample from the full distribution to create new clothes!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[109]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">generate_with_offset</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
    <span class="n">ev_mu_sl</span> <span class="o">=</span> <span class="n">ev_mu</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="n">ev_param_tensor</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">parametric</span><span class="p">(</span><span class="n">ev_mu_sl</span><span class="p">,</span><span class="n">ev_sigma</span><span class="p">)</span>
    <span class="n">ev_out</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">ev_param_tensor</span><span class="p">)</span><span class="o">.</span><span class="n">cpu</span><span class="p">()</span><span class="o">.</span><span class="n">detach</span><span class="p">()</span><span class="o">.</span><span class="n">numpy</span><span class="p">()</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">28</span><span class="p">,</span><span class="mi">28</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">ev_out</span>
    
<span class="n">generative_clothes</span> <span class="o">=</span> <span class="p">[</span><span class="n">generate_with_offset</span><span class="p">(</span><span class="n">i</span><span class="p">)</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="o">-.</span><span class="mi">5</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="o">.</span><span class="mi">25</span><span class="p">)]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[114]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="nb">len</span><span class="p">(</span><span class="n">generative_clothes</span><span class="p">),</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">20</span><span class="p">,</span><span class="mi">4</span><span class="p">))</span>
<span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">ax</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">axes</span><span class="o">.</span><span class="n">ravel</span><span class="p">()):</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">generative_clothes</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;bone_r&#39;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">&#39;off&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">suptitle</span><span class="p">(</span><span class="s1">&#39;sampling from latent space&#39;</span><span class="p">,</span> <span class="n">y</span><span class="o">=.</span><span class="mi">3</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>If you're running a live notebook, using the slider is quite an interesting experience!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[115]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">ipywidgets</span> <span class="k">import</span> <span class="n">interact</span><span class="p">,</span> <span class="n">interactive</span><span class="p">,</span> <span class="n">fixed</span><span class="p">,</span> <span class="n">interact_manual</span>
<span class="kn">import</span> <span class="nn">ipywidgets</span> <span class="k">as</span> <span class="nn">widgets</span>

<span class="k">def</span> <span class="nf">slide_f</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span><span class="mi">2</span><span class="p">))</span>
    <span class="n">ev_mu_sl</span> <span class="o">=</span> <span class="n">ev_mu</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="n">ev_param_tensor</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">parametric</span><span class="p">(</span><span class="n">ev_mu_sl</span><span class="p">,</span><span class="n">ev_sigma</span><span class="p">)</span>
    <span class="n">ev_out</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">ev_param_tensor</span><span class="p">)</span><span class="o">.</span><span class="n">cpu</span><span class="p">()</span><span class="o">.</span><span class="n">detach</span><span class="p">()</span><span class="o">.</span><span class="n">numpy</span><span class="p">()</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">28</span><span class="p">,</span><span class="mi">28</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">ev_out</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;bone_r&#39;</span><span class="p">)</span>


<span class="n">interact</span><span class="p">(</span><span class="n">slide_f</span><span class="p">,</span> <span class="n">x</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mf">3.0</span><span class="p">,</span> <span class="mf">0.02</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>





 
 
<div id="e043caec-7941-40bb-b1ea-28ce4599ab2f"></div>
<div class="output_subarea output_widget_view ">
<script type="text/javascript">
var element = $('#e043caec-7941-40bb-b1ea-28ce4599ab2f');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"model_id": "3589319e2aaf4bf0970952544e2ec667", "version_major": 2, "version_minor": 0}
</script>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[115]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;function __main__.slide_f(x)&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>And that's all for today!</p>

</div>
</div>
</div>
    </div>
  </div>
</body>

 


</html>
