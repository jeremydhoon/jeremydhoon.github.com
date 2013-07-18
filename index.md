---
layout: page
title: Nuts 'n Bolts
tagline: An unprincipled exploration of computing infrastructure.
---
{% include JB/setup %}

About
-----

I'm [Jeremy](https://github.com/jeremydhoon), an engineer with a passion
for the kind of software that helps others build phenomenal products and
services. Currenly, I build systems for storing lines of text at Stripe.
Previously, I built [systems for fighting spam](
https://www.facebook.com/notes/facebook-engineering/
fighting-spam-with-pure-functions/10151254986618920) at Facebook,
I helped teach machine learning under [David Parkes](http://www.eecs.harvard.edu/~parkes/), and interned at [Fog Creek Software](http://www.fogcreek.com/).

Posts
-----
<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>


