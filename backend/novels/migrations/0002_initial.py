# Generated by Django 4.1.7 on 2023-03-27 04:10

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('novels', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='novellike',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='novelcontentimage',
            name='novel_content',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='novels.novelcontent'),
        ),
        migrations.AddField(
            model_name='novelcontent',
            name='novel',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='novels.novel'),
        ),
        migrations.AddField(
            model_name='novelcomment',
            name='author',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='novelcomment',
            name='novel',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='novels.novel'),
        ),
        migrations.AddField(
            model_name='novel',
            name='author',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddConstraint(
            model_name='novellike',
            constraint=models.UniqueConstraint(fields=('novel', 'user'), name='unique_novel_user'),
        ),
        migrations.AddConstraint(
            model_name='novelcontent',
            constraint=models.UniqueConstraint(fields=('novel', 'step'), name='unique_novel_step'),
        ),
    ]