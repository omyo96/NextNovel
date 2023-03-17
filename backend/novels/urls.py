from django.urls import path

from novels.views import NovelPreviewAPI

app_name = 'novels'

urlpatterns = [
    path('<int:novel_id>/preview/', NovelPreviewAPI.as_view(), name='novel'),
]
