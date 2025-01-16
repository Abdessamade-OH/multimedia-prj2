import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'removeFirstLetter',
  standalone: true
})
export class RemoveFirstLetterPipe implements PipeTransform {

  transform(value: string): string {
    // Check if value is not empty and remove the first letter
    return value && value.length > 0 ? value.slice(1) : value;
  }

}
