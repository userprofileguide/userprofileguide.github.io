---
layout: archive
---

### {{ site.data.ui-text[site.locale].recent_posts | default: "Recent Posts" }}

{% for post in paginator.posts %}
  {% include archive-single.html %}
{% endfor %}

{% include paginator.html %}

### <h3  class="archive__subtitle">Current Research</h3>
